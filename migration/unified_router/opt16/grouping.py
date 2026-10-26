"""Offline response-metric trees; GPU assignment and stable counting permutation.

For Q the metric is covariance(K), and conversely for K. Distances therefore
approximate differences of centered attention logits, not Euclidean Q/K distance.
This is an initialization proxy, NOT a trained predictor of attention error.
"""
from dataclasses import dataclass
import torch
import triton as tr
import triton.language as tl

@dataclass
class Tree:
    weight: torch.Tensor
    threshold: torch.Tensor
    depth: int

@torch.no_grad()
def fit_tree(x, opposite, depth=6, rank=16, samples=2048):
    h, n, d = x.shape
    weights = torch.zeros((h, 2**depth-1, d), device=x.device)
    thresholds = torch.zeros((h, 2**depth-1), device=x.device)
    # Deterministic evenly spaced calibration samples. Caller owns split policy.
    xi = torch.linspace(0,n-1,min(n,samples),device=x.device).long()
    oi = torch.linspace(0,opposite.shape[1]-1,min(opposite.shape[1],samples),device=x.device).long()
    for head in range(h):
        xx=x[head,xi].float(); other=opposite[head,oi].float()
        other=other-other.mean(0)
        eigen, vec=torch.linalg.eigh(other.T@other/max(len(other),1))
        transform=vec[:,-rank:]*eigen[-rank:].clamp_min(1e-8).sqrt()[None]
        z=xx@transform
        queue=[(0,torch.arange(len(xx),device=x.device))]
        for level in range(depth):
            nxt=[]
            for node,ids in queue:
                if len(ids)<2:
                    # Empty/constant leaves remain deterministic; no duplication.
                    nxt.extend([(2*node+1,ids),(2*node+2,ids[:0])]);continue
                zz=z[ids]; centered=zz-zz.mean(0)
                cov=centered.T@centered
                direction=torch.zeros(rank,device=x.device);direction[cov.diag().argmax()]=1
                for _ in range(6):
                    direction=cov@direction;direction=direction/direction.norm().clamp_min(1e-12)
                plane=transform@direction;plane=plane/plane.norm().clamp_min(1e-12)
                response=xx[ids]@plane; threshold=response.median()
                weights[head,node]=plane;thresholds[head,node]=threshold
                right=response>threshold
                nxt.extend([(2*node+1,ids[~right]),(2*node+2,ids[right])])
            queue=nxt
    return Tree(weights.contiguous(),thresholds.contiguous(),depth)

@tr.jit
def _assign(X,W,T,L,N:tl.constexpr,D:tl.constexpr,XH:tl.constexpr,XN:tl.constexpr,DEPTH:tl.constexpr,BM:tl.constexpr):
    h=tl.program_id(1);i=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,D)
    x=tl.load(X+h*XH+i[:,None]*XN+d[None,:],i[:,None]<N,0).to(tl.float32)
    node=tl.full((BM,),0,tl.int32)
    for _ in range(DEPTH):
        w=tl.load(W+(h*((1<<DEPTH)-1)+node[:,None])*D+d[None,:])
        t=tl.load(T+h*((1<<DEPTH)-1)+node)
        right=tl.sum(x*w,1)>t
        node=2*node+1+right.to(tl.int32)
    tl.store(L+h*N+i,node-((1<<DEPTH)-1),i<N)

def labels(x,tree):
    h,n,d=x.shape
    out=torch.empty((h,n),device=x.device,dtype=torch.int32)
    _assign[(tr.cdiv(n,32),h)](x,tree.weight,tree.threshold,out,n,d,x.stride(0),x.stride(1),tree.depth,32,num_warps=4)
    return out

@tr.jit
def _local(L,S,R,H,N:tl.constexpr,C:tl.constexpr,TILES:tl.constexpr,TILE:tl.constexpr):
    tile=tl.program_id(0);head=tl.program_id(1);r=tl.arange(0,TILE);i=tile*TILE+r
    label=tl.load(L+head*N+i,i<N,C)
    key=label.to(tl.int32)*(N+TILE)+i
    ordered=tl.sort(key,descending=False);lab=ordered//(N+TILE);idx=ordered%(N+TILE)
    _,running=tl.associative_scan((lab,tl.full((TILE,),1,tl.int32)),0,_run_count)
    tl.store(S+(head*TILES+tile)*TILE+r,idx)
    tl.store(R+(head*TILES+tile)*TILE+r,running-1)
    c=tl.arange(0,C)
    counts=tl.sum((lab[:,None]==c[None,:]).to(tl.int32),0)
    tl.store(H+(head*TILES+tile)*C+c,counts)

@tr.jit
def _run_count(la,ca,lb,cb):return lb,tl.where(la==lb,ca+cb,cb)

@tr.jit
def _offset(H,O,TILES:tl.constexpr,C:tl.constexpr,TT:tl.constexpr):
    head=tl.program_id(0);bucket=tl.program_id(1);t=tl.arange(0,TT);c=tl.arange(0,C)
    hist=tl.load(H+(head*TILES+t[:,None])*C+c[None,:],t[:,None]<TILES,0)
    base=tl.sum(tl.sum(tl.where(c[None,:]<bucket,hist,0),1),0)
    column=tl.load(H+(head*TILES+t)*C+bucket,t<TILES,0)
    prefix=tl.cumsum(column,0)-column
    tl.store(O+(head*TILES+t)*C+bucket,prefix+base,t<TILES)

@tr.jit
def _scatter(L,S,R,O,P,N:tl.constexpr,C:tl.constexpr,TILES:tl.constexpr,TILE:tl.constexpr):
    tile=tl.program_id(0);head=tl.program_id(1);r=tl.arange(0,TILE)
    i=tl.load(S+(head*TILES+tile)*TILE+r)
    lab=tl.load(L+head*N+i,i<N,0);rank=tl.load(R+(head*TILES+tile)*TILE+r)
    base=tl.load(O+(head*TILES+tile)*C+lab)
    tl.store(P+head*N+base+rank,i,i<N)

def permutation(lab,buckets):
    assert lab.dtype==torch.int32 and buckets&(buckets-1)==0
    h,n=lab.shape;t=tr.cdiv(n,128)
    local=torch.empty((h,t,128),device=lab.device,dtype=torch.int32);rank=torch.empty_like(local)
    hist=torch.empty((h,t,buckets),device=lab.device,dtype=torch.int32);offset=torch.empty_like(hist)
    out=torch.empty_like(lab)
    _local[(t,h)](lab,local,rank,hist,n,buckets,t,128,num_warps=4)
    _offset[(h,buckets)](hist,offset,t,buckets,tr.next_power_of_2(t),num_warps=4)
    _scatter[(t,h)](lab,local,rank,offset,out,n,buckets,t,128,num_warps=4)
    return out

def order(x,tree):return permutation(labels(x,tree),2**tree.depth)

def gather(x,index):
    return x.gather(1,index.long()[...,None].expand(-1,-1,x.shape[-1]))

def restore(x,index):
    out=torch.empty_like(x);out.scatter_(1,index.long()[...,None].expand_as(x),x);return out
