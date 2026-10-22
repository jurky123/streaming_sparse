"""Matched Q-before/Q-after branches; exactly one shared frontend and partition."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..v03.model import PairMLP, PairLinear, BlockAggregator, SetEncoderLayer
from ..router.clustering import soft_kmeans
from ..router.budget_selection import budget_mask


@dataclass
class RouterConfig:
    version: str = '0.4'
    layers: tuple = (4,14,24)
    heads: tuple = tuple(range(12))
    residual_rank: int = 1
    target_size: int = 64
    max_blocks: int = 256
    cluster_iterations: int = 3
    tau_cluster: float = .1
    tau_mask: float = .2
    max_size_factor: float = 4.
    bf16: bool = True

    def __post_init__(self):
        self.layers,self.heads=tuple(self.layers),tuple(self.heads)
        if self.version!='0.4' or self.layers!=(4,14,24) or self.heads!=tuple(range(12)):
            raise ValueError('v0.4 fixes layers4/14/24 and all12 heads')
        if self.residual_rank!=1 or self.target_size<1 or self.max_blocks<1 or self.cluster_iterations<1:
            raise ValueError('Invalid clustering parameters / changed conditioning')
        if min(self.tau_cluster,self.tau_mask)<=0 or self.max_size_factor<1:
            raise ValueError('Invalid temperatures or repair bound')


def partition_hash(hard):
    labels=hard.detach().argmax(-1).to(device='cpu',dtype=torch.int64).contiguous()
    sizes=hard.detach().sum(0).to(device='cpu',dtype=torch.int64).contiguous()
    h=hashlib.sha256()
    for x in (labels,sizes,torch.arange(len(labels),dtype=torch.int64)):
        h.update(x.numpy().tobytes())
    return h.hexdigest()


class SharedFrontend(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.config=c
        pairs=len(c.layers)*len(c.heads)
        self.features=PairMLP((256,256,128,128),pairs,c.residual_rank)
        self.aggregator=BlockAggregator(pairs,c.residual_rank)

    def forward(self,k,v,layer,head):
        if k.ndim!=2 or k.shape!=v.shape or k.shape[1]!=128 or len(k)==0:
            raise ValueError('Expected nonempty Nx128 raw history KV')
        c=self.config
        pair=c.layers.index(int(layer))*len(c.heads)+c.heads.index(int(head))
        # FP32 island: unchanged clustering / repair / mean-max-min arithmetic.
        with torch.autocast(device_type=k.device.type,enabled=False):
            features=self.features(torch.cat((k.float(),v.float()),-1),pair)
            assignment,hard,p,usage=soft_kmeans(features,temperature=c.tau_cluster,
                iterations=c.cluster_iterations,target_size=c.target_size,max_blocks=c.max_blocks,
                max_size_factor=c.max_size_factor)
            z,empty=self.aggregator(features,assignment,hard,pair)
        return dict(features=features,assignment=assignment,hard=hard,probabilities=p,
                    raw_labels=p.argmax(-1),usage_diagnostic=usage,z=z,empty=empty,pair=pair,
                    layer_id=int(layer),head_id=int(head))


class OrderBranch(nn.Module):
    def __init__(self,c,order):
        super().__init__()
        self.order=order
        pairs=len(c.layers)*len(c.heads)
        self.query=PairMLP((128,256,128),pairs,c.residual_rank)
        self.fusion=PairLinear(384,128,pairs,c.residual_rank)
        self.interaction=nn.ModuleList(SetEncoderLayer(pairs,c.residual_rank) for _ in range(2))
        self.score=PairLinear(128,1,pairs,c.residual_rank)

    def interact(self,z,empty,pair):
        if z.device.type=='cuda' and torch.is_autocast_enabled():
            z=z.to(torch.bfloat16)
        for layer in self.interaction:
            z=layer(z,empty,pair)
        return z

    def prepare(self,z,empty,pair):
        # This API deliberately cannot receive Q or any Q-derived value.
        return self.interact(z,empty,pair) if self.order=='after' else z

    def forward(self,q,context,empty,pair):
        if context.device.type=='cuda' and torch.is_autocast_enabled():
            context=context.to(torch.bfloat16)
        query=self.query(q.float().mean(0),pair).expand_as(context)
        x=F.gelu(self.fusion(torch.cat((context,query,context*query),-1),pair))
        if self.order=='before':
            x=self.interact(x,empty,pair)
        # Keep ranking resolution when BF16 hidden states have a common component.
        with torch.autocast(device_type=x.device.type,enabled=False):
            return self.score(x.float(),pair).squeeze(-1).masked_fill(empty,0)


class PairedRouter(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or RouterConfig()
        self.frontend=SharedFrontend(self.config)
        before=OrderBranch(self.config,'before')
        after=deepcopy(before)
        after.order='after'
        self.branches=nn.ModuleDict(dict(A=before,B=after))
        self._assert_capacity()

    def _amp(self,device):
        return torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                              enabled=self.config.bf16 and device.type=='cuda')

    def _assert_capacity(self):
        assert sum(p.numel() for p in self.branches['A'].parameters())==sum(p.numel() for p in self.branches['B'].parameters())

    def precompute(self,k,v,positions=None,head_id=0,layer_id=4):
        state=self.frontend(k,v,layer_id,head_id)
        with self._amp(k.device):
            state['contexts']={name:b.prepare(state['z'],state['empty'],state['pair'])
                               for name,b in self.branches.items()}
        state['partition_hash']=partition_hash(state['hard'])
        return state

    def route(self,q,state,keep):
        if q.ndim!=2 or q.shape[-1]!=128 or not 0<len(q)<=16 or not 0<=keep<=1:
            raise ValueError('Expected one query tile and keep in [0,1]')
        with self._amp(q.device):
            scores={name:b(q,state['contexts'][name],state['empty'],state['pair'])
                    for name,b in self.branches.items()}
        budget=math.floor(keep*len(state['hard']))
        routes={}
        with torch.autocast(device_type=q.device.type,enabled=False):
            for name,s in scores.items():
                mask,hard,soft=budget_mask(s,state['assignment'],state['hard'],budget,
                                           temperature=self.config.tau_mask)
                assert int((hard*state['hard'].sum(0)).sum())<=budget
                routes[name]=dict(mask=mask,hard_mask=hard,soft_mask=soft,scores=s,budget=budget)
        return routes

    def capacity(self):
        front=sum(p.numel() for p in self.frontend.parameters())
        branches={name:sum(p.numel() for p in b.parameters()) for name,b in self.branches.items()}
        return dict(shared_frontend=front,branches=branches,inference_per_arm={k:front+v for k,v in branches.items()},
                    paired_training_total=front+sum(branches.values()))

    def checkpoint(self):
        return dict(config=asdict(self.config),router=self.state_dict(),capacity=self.capacity())


def load_pair(path,device='cpu'):
    value=torch.load(path,map_location=device,weights_only=True)
    model=PairedRouter(RouterConfig(**value['config'])).to(device)
    model.load_state_dict(value['router'],strict=True)
    return model,value
