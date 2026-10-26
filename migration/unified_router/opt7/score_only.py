"""Score-only ablation: independent query/page decisions, no learned AV correction.

This changes routing semantics and must not be presented as equivalent to Opt6.
Only page scores and the final mask are materialized; prototype logits stay on chip.
"""
import math
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl
from ..opt4.network import CompactCache
from ..opt2.network import load_optimized

@tr.jit
def _score(Q,K,LC,S,N:tl.constexpr,B:tl.constexpr,QM:tl.constexpr,GP:tl.constexpr):
    h=tl.program_id(2);qi=tl.program_id(0)*QM+tl.arange(0,QM)
    pg=tl.program_id(1)*GP+tl.arange(0,GP)
    proto=tl.arange(0,GP*16);pages=tl.program_id(1)*GP+proto//16
    d=tl.arange(0,64);p=tl.arange(0,16)
    q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0.)
    key=tl.load(K+((h*B+pages[None,:])*16+proto[None,:]%16)*64+d[:,None],pages[None,:]<B,0.)
    raw=tl.dot(q,key).reshape((QM,GP,16))
    lc=tl.load(LC+(h*B+pg[:,None])*16+p[None,:],pg[:,None]<B,float('-inf'))
    logits=raw/math.sqrt(128)+lc[None,:,:]
    maximum=tl.max(logits,2);maximum=tl.where(pg[None,:]<B,maximum,0.)
    score=maximum+tl.log(tl.sum(tl.exp(logits-maximum[:,:,None]),2))
    tl.store(S+(h*N+qi[:,None])*B+pg[None,:],score,(qi[:,None]<N)&(pg[None,:]<B))

@tr.jit
def _threshold(S,M,B:tl.constexpr,P:tl.constexpr,BETA:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,P)
    x=tl.load(S+row*B+p,p<B,0.)
    mean=tl.sum(x,0)/B
    delta=tl.where(p<B,x-mean,0.);std=tl.sqrt(tl.sum(delta*delta,0)/B)
    mask=(p<B)&((x>mean+BETA*std)|(std<=1.e-6))
    best=tl.max(tl.where(p<B,x,float('-inf')),0)
    index=tl.min(tl.where((p<B)&(x==best),p,2147483647),0)
    mask=mask|((tl.sum(mask.to(tl.int32),0)==0)&(p==index))
    tl.store(M+row*B+p,mask,p<B)

def pack(prepared):
    keys=[];counts=[]
    for key,lc,_ in prepared[1]:
        h,b,p=lc.shape
        keys.append(F.pad(key.reshape(h,b,p,64),(0,0,0,16-p)).half())
        counts.append(F.pad(lc,(0,16-p),value=-float('inf')))
    return torch.cat(keys,1),torch.cat(counts,1)

class ScoreRoute:
    def __init__(self,qm=64,gp=4):
        self.qm=qm;self.gp=gp;self.pack=torch.compile(pack,fullgraph=True,dynamic=False)
    def begin(self,prepared,sketch):
        return (prepared[0],*self.pack(prepared))
    def run(self,layer,q,sketch,state,step,router):
        basis,key,lc=state;h,n,_=q.shape;b=lc.shape[1]
        qq=torch.bmm(q.float(),basis).half()
        score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);mask=torch.empty_like(score,dtype=torch.bool)
        _score[(tr.cdiv(n,self.qm),tr.cdiv(b,self.gp),h)](qq,key,lc,score,n,b,self.qm,self.gp,num_warps=4,num_stages=1)
        _threshold[(h*n,)](score,mask,b,tr.next_power_of_2(b),router.beta,num_warps=4,enable_fp_fusion=False)
        return mask

def load_score(directory,training_seed=0,**options):
    net,m=load_optimized(directory,training_seed,**options)
    route=ScoreRoute();net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
    return net,m

def main():
    from .. import fast_bench
    fast_bench.load_fast=load_score;fast_bench.main()
if __name__=='__main__':main()
