"""Shared historical cache; matched per-query relevance/correction comparisons."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..v03.model import PairMLP, PairLinear, PairNorm, SetEncoderLayer
from ..v04.model import SharedFrontend, partition_hash
from .primitives import budget_mask


@dataclass
class RouterConfig:
    version: str = '0.5'
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
    relation_dim: int = 64
    center_relations: bool = True
    query_log_norm: bool = True

    def __post_init__(self):
        self.layers,self.heads=tuple(self.layers),tuple(self.heads)
        if self.version!='0.5' or self.layers!=(4,14,24) or self.heads!=tuple(range(12)):
            raise ValueError('Expected v0.5 with all36 layer/head pairs')
        if self.residual_rank!=1 or self.relation_dim!=64 or min(self.target_size,self.max_blocks,self.cluster_iterations)<1:
            raise ValueError('Invalid architecture/clustering configuration')
        if min(self.tau_cluster,self.tau_mask)<=0 or self.max_size_factor<1:
            raise ValueError('Invalid gate/clustering temperature or repair bound')


class HistoricalFrontend(nn.Module):
    def __init__(self,c):
        super().__init__()
        pairs=len(c.layers)*len(c.heads)
        self.kv=SharedFrontend(c)
        self.z_norm=PairNorm(128,pairs)
        self.pre=SetEncoderLayer(pairs,c.residual_rank)

    def forward(self,k,v,layer,head):
        state=self.kv(k,v,layer,head)
        z=self.z_norm(state['z'],state['pair']).masked_fill(state['empty'][:,None],0)
        with torch.autocast(device_type=k.device.type,dtype=torch.bfloat16,
                            enabled=self.kv.config.bf16 and k.device.type=='cuda'):
            h=self.pre(z.to(torch.bfloat16) if k.device.type=='cuda' and self.kv.config.bf16 else z,
                       state['empty'],state['pair'])
        state.update(z_normalized=z,h=h,block_ids=torch.arange(len(z),device=k.device),
                     partition_hash=partition_hash(state['hard']))
        return state


class QueryFusion(nn.Module):
    def __init__(self,c):
        super().__init__()
        pairs=len(c.layers)*len(c.heads);rank=c.residual_rank
        self.query=PairMLP((128,256,128),pairs,rank)
        self.query_norm=PairNorm(128,pairs)
        self.history_norm=PairNorm(128,pairs)
        self.norm_projection=PairLinear(1,128,pairs,rank) if c.query_log_norm else None
        self.fusion=PairMLP((384,256,128),pairs,rank)
        self.scale=nn.Parameter(torch.tensor(.1))
        self.output_norm=PairNorm(128,pairs)

    def forward(self,q,h,empty,pair):
        # Q is never pooled: all operations retain its independent leading axis.
        e=self.query_norm(self.query(q.float(),pair),pair)
        if self.norm_projection is not None:
            e=e+self.norm_projection(q.float().norm(dim=-1,keepdim=True).clamp_min(1e-8).log(),pair)
        hn=self.history_norm(h,pair)
        hh=hn[None].expand(len(q),-1,-1);ee=e[:,None].expand(-1,len(h),-1)
        x=h[None]+self.scale*self.fusion(torch.cat((hh,ee,hh*ee),-1),pair)
        return self.output_norm(x,pair).masked_fill(empty[None,:,None],0),e


class InteractionBlock(nn.Module):
    def __init__(self,c,signed):
        super().__init__()
        self.signed=signed;self.center=c.center_relations
        pairs=len(c.layers)*len(c.heads);rank=c.residual_rank
        self.norm1,self.norm2=PairNorm(128,pairs),PairNorm(128,pairs)
        self.out=PairLinear(128,128,pairs,rank)
        self.ffn=PairMLP((128,512,128),pairs,rank)
        self.scale=nn.Parameter(torch.tensor(.1))
        if signed:
            self.u=PairMLP((128,128,c.relation_dim),pairs,rank)
            self.u_norm=PairNorm(c.relation_dim,pairs)
            self.value=PairLinear(128,128,pairs,rank)
        else:
            self.qkv=PairLinear(128,384,pairs,rank)

    def forward(self,x,empty,pair,mode='normal',trace=False):
        normalized=self.norm1(x,pair)
        relation=None
        if self.signed:
            # All signed geometry, normalization and aggregation use FP32.
            with torch.autocast(device_type=x.device.type,enabled=False):
                u=self.u_norm(self.u(normalized.float(),pair),pair)
                valid=(~empty).float()[None,:,None]
                if self.center and mode!='no_center':
                    u=u-(u*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
                u=F.normalize(u,dim=-1,eps=1e-6)*valid
                relation=(u@u.transpose(-1,-2)).clamp(-1,1)
                allowed=(~empty)[:,None]&(~empty)[None,:]&~torch.eye(len(empty),device=x.device,dtype=torch.bool)
                relation=relation.masked_fill(~allowed[None],0)
                if mode=='positive':relation=relation.clamp_min(0)
                elif mode=='absolute':relation=relation.abs()
                elif mode=='zero':relation=relation*0
                elif mode not in ('normal','no_center'):raise ValueError(mode)
                value=self.value(normalized.float(),pair)
                message=(relation@value)/(relation.abs().sum(-1,keepdim=True)+1e-6)
                projected=self.out(message,pair)-self.out(torch.zeros_like(message),pair)
        else:
            q,k,v=self.qkv(normalized,pair).chunk(3,-1)
            q,k,v=[a.reshape(len(x),len(empty),4,32).transpose(1,2) for a in (q,k,v)]
            message=F.scaled_dot_product_attention(q,k,v,attn_mask=(~empty)[None,None,None,:],dropout_p=0.)
            message=message.transpose(1,2).reshape_as(x)
            projected=self.out(message,pair)-self.out(torch.zeros_like(message),pair)
        update=self.scale*projected
        y=x+update
        y=y+self.ffn(self.norm2(y,pair),pair)
        y=y.masked_fill(empty[None,:,None],0)
        info=dict(update=update,message=message)
        if trace:info.update(relation=relation,interaction_vector=u if self.signed else None)
        return y,projected,info


class ScoreBranch(nn.Module):
    def __init__(self,c,kind):
        super().__init__()
        self.kind=kind
        pairs=len(c.layers)*len(c.heads);rank=c.residual_rank
        self.interaction=nn.ModuleList(InteractionBlock(c,kind=='R3') for _ in range(2))
        self.score_norm=PairNorm(128,pairs)
        if kind=='R1':
            self.score=PairLinear(128,1,pairs,rank)
        else:
            self.size=PairLinear(1,128,pairs,rank)
            self.relevance=PairMLP((128,128,1),pairs,rank)
            self.message_norm=PairNorm(128,pairs)
            self.correction=PairMLP((256,128,1),pairs,rank)
            self.correction_scale=nn.Parameter(torch.tensor(.1))

    def forward(self,q,state,fused,mode='normal',trace=False):
        pair,empty=state['pair'],state['empty']
        x,e=fused
        y=x;messages=[];infos=[]
        for layer in self.interaction:
            y,m,info=layer(y,empty,pair,mode,trace)
            messages.append(m);infos.append(info)
        # FP32 scores preserve fine ranking differences; original KV is never normalized.
        with torch.autocast(device_type=q.device.type,enabled=False):
            if self.kind=='R1':
                relevance=self.score(self.score_norm(y.float(),pair),pair).squeeze(-1)
                delta=torch.zeros_like(relevance)
                scores=relevance
            else:
                size=state['hard'].sum(0).log1p()[:,None]
                relevance=self.relevance(x.float()+self.size(size,pair)[None],pair).squeeze(-1)
                message=torch.stack(messages).mean(0).float()
                # Subtraction enforces correction(x,0)=0 even after affine biases learn.
                zero=torch.zeros_like(message)
                mn=self.message_norm(message,pair)-self.message_norm(zero,pair)
                yn=self.score_norm(y.float(),pair)
                delta=(self.correction(torch.cat((yn,mn),-1),pair)-
                       self.correction(torch.cat((yn,torch.zeros_like(mn)),-1),pair)).squeeze(-1)
                scores=relevance+self.correction_scale*delta
        result=dict(scores=scores.masked_fill(empty[None],0),relevance=relevance,correction=delta)
        if trace:result.update(fusion=x,query_embedding=e,hidden=y,layers=infos)
        return result


class Router(nn.Module):
    def __init__(self,config=None,arms=('R1','R2','R3')):
        super().__init__()
        self.config=config or RouterConfig()
        self.frontend=HistoricalFrontend(self.config)
        self.query_fusion=QueryFusion(self.config)
        self.arms=tuple(arms)
        if not self.arms or not set(self.arms)<={'R1','R2','R3'}:raise ValueError(arms)
        templates={kind:ScoreBranch(self.config,kind) for kind in ('R1','R2','R3')}
        # Match all compatible initial tensors (fusion, norms, heads, FFNs/out) exactly.
        reference=templates['R2'].state_dict()
        for kind in ('R1','R3'):
            own=templates[kind].state_dict()
            own.update({k:v for k,v in reference.items() if k in own and own[k].shape==v.shape})
            templates[kind].load_state_dict(own)
        self.branches=nn.ModuleDict({kind:templates[kind] for kind in self.arms})

    def precompute(self,k,v,positions=None,head_id=0,layer_id=4):
        return self.frontend(k,v,layer_id,head_id)

    def predict(self,q,state,trace=False,intervention='normal',only=None):
        if q.ndim!=2 or q.shape[1]!=128 or not 1<=len(q)<=16:
            raise ValueError('Expected 1..16 independent queries of width128')
        results={}
        with torch.autocast(device_type=q.device.type,dtype=torch.bfloat16,
                            enabled=self.config.bf16 and q.device.type=='cuda'):
            fused=self.query_fusion(q,state['h'],state['empty'],state['pair'])
        for name,branch in self.branches.items():
            if only is not None and name not in only:continue
            with torch.autocast(device_type=q.device.type,dtype=torch.bfloat16,
                                enabled=self.config.bf16 and q.device.type=='cuda'):
                result=branch(q,state,fused,intervention if name=='R3' else 'normal',trace)
            results[name]=result
        return results

    def route(self,q,state,keep=.2,trace=False,intervention='normal',predictions=None,only=None):
        if not 0<=keep<=1:raise ValueError(keep)
        budget=math.floor(keep*len(state['hard']))
        results={}
        predictions=self.predict(q,state,trace,intervention,only) if predictions is None else predictions
        for name,prediction in predictions.items():
            result=dict(prediction)
            with torch.autocast(device_type=q.device.type,enabled=False):
                mask,hard,soft=budget_mask(result['scores'].float(),state['assignment'],state['hard'],budget,
                                           self.config.tau_mask,state['block_ids'])
            result.update(mask=mask,hard_mask=hard,soft_mask=soft,budget=budget)
            results[name]=result
        return results

    def capacity(self):
        front=sum(p.numel() for p in self.frontend.parameters())
        query=sum(p.numel() for p in self.query_fusion.parameters())
        arms={name:sum(p.numel() for p in arm.parameters()) for name,arm in self.branches.items()}
        return dict(shared_frontend=front,shared_query_fusion=query,branches=arms,
                    per_model={k:front+query+v for k,v in arms.items()},joint_total=front+query+sum(arms.values()),
                    R3_over_R2=(front+query+arms.get('R3',0))/max(front+query+arms.get('R2',0),1))

    def checkpoint(self):
        return dict(config=asdict(self.config),arms=self.arms,router=self.state_dict(),capacity=self.capacity())


def load_router(path,device='cpu'):
    value=torch.load(path,map_location=device,weights_only=True)
    model=Router(RouterConfig(**value['config']),value['arms']).to(device)
    model.load_state_dict(value['router'],strict=True)
    return model,value
