"""IO-balanced alternative: keep FP16 weights, never emit prototype logits."""
import torch
import triton as tr
import triton.language as tl
from .route import FlashRoute
from .kernels import _score_energy
from ..opt4_fused import means
from ..opt4.kernels import _gram_stats
from .prepare import prepare_weights

@tr.jit
def _weighted_candidates(W,V,START,COUNT,C,REM,ADD,CV,N:tl.constexpr,R:tl.constexpr):
    row=tl.program_id(0);h=row//N;q=row%N
    slot=tl.arange(0,16);p=tl.arange(0,16);d=tl.arange(0,32)
    i=tl.load(REM+row*8+slot,slot<8,-1);j=tl.load(ADD+row*8+slot-8,slot>=8,-1)
    page=tl.where(slot<8,i,j);valid=page>=0
    start=tl.load(START+page,valid,0);count=tl.load(COUNT+page,valid,0)
    cost=tl.load(C+page,valid,1.)
    w=tl.load(W+(h*R+start[:,None]+p[None,:])*N+q,valid[:,None]&(p[None,:]<count[:,None]),0.).to(tl.float32)/cost[:,None]
    v=tl.load(V+(h*R+start[:,None,None]+p[None,:,None])*32+d[None,None,:],valid[:,None,None]&(p[None,:,None]<count[:,None,None]),0.)
    content=tl.sum(w[:,:,None]*v,1)
    tl.store(CV+(row*16+slot[:,None])*32+d[None,:],content)

class WeightRoute(FlashRoute):
    def __init__(self,**options):
        super().__init__(**options)
        self.prepare_weights=torch.compile(prepare_weights,fullgraph=True,dynamic=False)

    def begin(self,prepared,sketch):
        state=dict(prepared=prepared,costs=sketch.costs)
        if prepared[1][0][2]is not None:state.update(self.prepare_weights(prepared,self.energy=='content'))
        return state

    def score(self,q,state):
        h,n,_=q.shape;b=state['costs'].numel();r=state['value'].shape[1]
        qq=torch.bmm(q.float(),state['prepared'][0]).half()
        score=torch.empty((h,n,b),device=q.device,dtype=torch.float32);energy=torch.empty_like(score)
        weights=torch.empty((h,r,n),device=q.device,dtype=torch.float16)
        _score_energy[(tr.cdiv(n,self.qm),tr.cdiv(b,self.page_group),h)](qq,state['padded_key'],state['padded_lc'],state['grams'],state['padded_value'],
            score,energy,n,b,self.qm,self.page_group,self.energy=='content',weights,state['starts'],state['counts'],state['costs'],r,True,
            num_warps=4,num_stages=1,enable_fp_fusion=True)
        return score,(weights,energy)

    def stats(self,q,sketch,score,mask,rem,add,raw,state):
        weights,energy=raw;h,n,b=score.shape;r=weights.shape[1]
        stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
        cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
        mean=means(weights,mask,state,'fp16')
        _weighted_candidates[(h*n,)](weights,state['value'],state['starts'],state['counts'],sketch.costs,rem,add,cv,n,r,
            num_warps=4,enable_fp_fusion=True)
        _gram_stats[(h*n,)](mean,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=4,enable_fp_fusion=False)
        return stats,cv
