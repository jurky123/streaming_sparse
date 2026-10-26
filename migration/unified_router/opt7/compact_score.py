"""Score-only router without padding history P2/P4/P8 to P16."""
import math
import torch
import triton as tr
import triton.language as tl
from .score_only import _threshold,load_score
from ..opt4.network import CompactCache

@tr.jit
def _score_part(Q,K,LC,S,N:tl.constexpr,B:tl.constexpr,TOTAL:tl.constexpr,OFFSET:tl.constexpr,
                P:tl.constexpr,QM:tl.constexpr,GP:tl.constexpr):
    h=tl.program_id(2);qi=tl.program_id(0)*QM+tl.arange(0,QM)
    pg=tl.program_id(1)*GP+tl.arange(0,GP)
    proto=tl.arange(0,GP*P);pages=tl.program_id(1)*GP+proto//P
    d=tl.arange(0,64);p=tl.arange(0,P)
    q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0.)
    key=tl.load(K+((h*B+pages[None,:])*P+proto[None,:]%P)*64+d[:,None],pages[None,:]<B,0.)
    raw=tl.dot(q,key).reshape((QM,GP,P))
    lc=tl.load(LC+(h*B+pg[:,None])*P+p[None,:],pg[:,None]<B,float('-inf'))
    logits=raw/math.sqrt(128)+lc[None,:,:]
    maximum=tl.max(logits,2);maximum=tl.where(pg[None,:]<B,maximum,0.)
    score=maximum+tl.log(tl.sum(tl.exp(logits-maximum[:,:,None]),2))
    tl.store(S+(h*N+qi[:,None])*TOTAL+OFFSET+pg[None,:],score,(qi[:,None]<N)&(pg[None,:]<B))

class CompactScore:
    def begin(self,prepared,sketch):return prepared[0],[(k.half(),lc)for k,lc,_ in prepared[1]]
    def run(self,layer,q,sketch,state,step,router):
        basis,parts=state;h,n,_=q.shape;b=sketch.costs.numel()
        qq=torch.bmm(q.float(),basis).half();score=torch.empty((h,n,b),device=q.device,dtype=torch.float32)
        offset=0
        for key,lc in parts:
            _,np,p=lc.shape;gp=64//p
            _score_part[(tr.cdiv(n,64),tr.cdiv(np,gp),h)](qq,key,lc,score,n,np,b,offset,p,64,gp,num_warps=4,num_stages=1)
            offset+=np
        mask=torch.empty_like(score,dtype=torch.bool)
        _threshold[(h*n,)](score,mask,b,tr.next_power_of_2(b),router.beta,num_warps=4,enable_fp_fusion=False)
        return mask

def load_compact_score(directory,training_seed=0,**options):
    net,m=load_score(directory,training_seed,**options)
    route=CompactScore();net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
    return net,m
