from dataclasses import asdict, dataclass
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..router.model import RouterConfig as V01Config
from ..router.joint_feature import JointFeature
from ..router.clustering import squared_distance, repair_labels
from ..router.aggregator import BlockAggregator
from ..router.encoder import BlockEncoder
from ..router.decoder import QueryDecoder
from ..router.budget_selection import budget_mask, priority, select_under_budget


@dataclass
class RouterConfig(V01Config):
    conditioned: bool = True
    encoder_type: str = 'transformer'
    version: str = '0.2'


class Features(JointFeature):
    def __init__(self, config):
        super().__init__(config.head_dim, config.heads, config.layers)
        self.conditioned = config.conditioned
        if not self.conditioned:
            self.head.requires_grad_(False)
            self.layer.requires_grad_(False)

    def forward(self, k, v, positions, head_id, layer_id):
        kv = torch.cat((k.float(),v.float()),-1)
        if self.conditioned:
            kv = kv+self.head.weight[head_id]+self.layer.weight[layer_id]
        kv = kv+self.position(positions.float()/64)
        return self.group(kv),self.content(kv)


class IndependentBlockEncoder(nn.Module):
    """Two residual per-block FFNs, parameter count close to two Transformer layers."""
    def __init__(self):
        super().__init__()
        self.norms=nn.ModuleList([nn.LayerNorm(256) for _ in range(2)])
        self.ffns=nn.ModuleList([nn.Sequential(nn.Linear(256,1536),nn.GELU(),nn.Linear(1536,256)) for _ in range(2)])
        self.norm=nn.LayerNorm(256)

    def forward(self,z,empty):
        for norm,ffn in zip(self.norms,self.ffns):
            z=z+ffn(norm(z))
        return self.norm(z).masked_fill(empty[:,None],0)


def cluster(features, config):
    n=len(features)
    b=min(config.max_blocks,math.ceil(n/config.target_size))
    centers=features[torch.linspace(0,n-1,b,device=features.device).long()]
    for _ in range(config.cluster_iterations):
        p=(-squared_distance(features,centers)/config.tau_cluster).softmax(-1)
        centers=p.T@features/p.sum(0)[:,None].clamp_min(1e-8)
    distance=squared_distance(features,centers)
    p=(-distance/config.tau_cluster).softmax(-1)
    raw=p.argmax(-1)
    labels=repair_labels(raw,distance.detach(),math.ceil(config.max_size_factor*n/b))
    hard=F.one_hot(labels,b).float()
    a=hard+(p-p.detach())
    usage=p.mean(0)
    reg=(usage*(usage.clamp_min(1e-8)*b).log()).sum()
    return a,hard,p,reg,raw


class SparseRouter(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or RouterConfig()
        c=self.config
        self.features=Features(c)
        self.aggregator=BlockAggregator()
        self.encoder=BlockEncoder() if c.encoder_type=='transformer' else IndependentBlockEncoder()
        self.decoder=QueryDecoder(c.head_dim)
        # Freeze the complete grouping input map, not just its last MLP: otherwise
        # trainable head/layer/position embeddings would silently change partitions.
        self.frozen_group_features=copy.deepcopy(self.features).requires_grad_(False) if c.grouping=='frozen' else None
        if c.grouping in ('frozen','spatial'):
            self.features.group.requires_grad_(False)

    def precompute(self,k,v,positions,head_id=0,layer_id=0,assignment_override=None):
        g,content=self.features(k,v,positions,head_id,layer_id)
        raw=None
        if assignment_override is not None:
            a=hard=assignment_override
            p=hard
            reg=g.sum()*0
        elif self.config.grouping=='spatial':
            from ..evaluation.baselines import spatial_assignment
            a=hard=spatial_assignment(positions,self.config.target_size).to(k.device)
            p=hard
            reg=g.sum()*0
        else:
            if self.frozen_group_features is not None:
                with torch.no_grad():
                    g,_=self.frozen_group_features(k,v,positions,head_id,layer_id)
            a,hard,p,reg,raw=cluster(g,self.config)
        z,empty=self.aggregator(content,a,hard,positions)
        h=self.encoder(z,empty)
        return dict(assignment=a,hard=hard,probabilities=p,h=h,regularizer=reg,raw_labels=raw)

    def route(self,q,state,keep_ratio,relaxation=True):
        if not 0<=keep_ratio<=1:
            raise ValueError('Invalid keep ratio')
        scores=self.decoder(q,state['h'])
        budget=math.floor(keep_ratio*state['hard'].shape[0])
        if relaxation:
            m,hard,soft=budget_mask(scores,state['assignment'],state['hard'],budget,self.config.alpha,self.config.tau_mask)
        else:
            costs=state['hard'].sum(0)
            hard=select_under_budget(priority(scores,costs,self.config.alpha),costs,budget)
            m,soft=hard,None
        return dict(mask=m,hard_mask=hard,soft_mask=soft,budget=budget,scores=scores)

    def checkpoint(self):
        return dict(config=asdict(self.config),router=self.state_dict())


def load_router(path,device='cpu'):
    value=torch.load(path,map_location=device,weights_only=True)
    config=dict(value['config'])
    config.setdefault('version','0.2')
    model=SparseRouter(RouterConfig(**config)).to(device)
    model.load_state_dict(value['router'])
    return model
