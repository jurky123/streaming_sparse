"""Initial fixed-budget selection with deterministic low-index tie handling.
Native torch.topk does not specify stable ties; ties may select different pages.
Finite sentinels are required: this Triton sort uses arithmetic masking, so
Inf * 0 would create NaNs during compare-and-swap. Inputs are probability masses.
"""
import torch,triton as tr,triton.language as tl
@tr.jit
def _pick(M,MASK,IDS,P:tl.constexpr,T:tl.constexpr,KEEP:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,B);own=row%T+P-T
    mass=tl.load(M+row*P+p,p<P,-1.e30);mass=tl.where(p==own,1.e30,mass)
    ordered=tl.sort(mass,descending=True)
    cutoff=tl.sum(tl.where(p==KEEP-1,ordered,0.),0)
    above=(mass>cutoff)&(p<P);tie=(mass==cutoff)&(p<P)
    needed=KEEP-tl.sum(above.to(tl.int32),0);tie_rank=tl.cumsum(tie.to(tl.int32),0)
    selected=above|(tie&(tie_rank<=needed));rank=tl.cumsum(selected.to(tl.int32),0)-1
    tl.store(MASK+row*P+p,selected,p<P)
    tl.store(IDS+row*KEEP+rank,p,selected&(rank<KEEP))
def pick(mass,keep):
    h,t,p=mass.shape;assert 1<=keep<p and t<=p and mass.is_contiguous()
    mask=torch.empty_like(mass,dtype=torch.bool);ids=torch.empty(h,t,keep,device=mass.device,dtype=torch.int64)
    _pick[(h*t,)](mass,mask,ids,p,t,keep,tr.next_power_of_2(p),num_warps=4)
    return mask,ids
