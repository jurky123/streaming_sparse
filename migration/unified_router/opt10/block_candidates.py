import triton as tr
import triton.language as tl
@tr.jit
def _weighted_candidates(W,V,START,COUNT,C,REM,ADD,CV,N:tl.constexpr,R:tl.constexpr,QT:tl.constexpr):
    row=tl.program_id(0);h=row//N;q=row%N
    slot=tl.arange(0,16);p=tl.arange(0,16);d=tl.arange(0,32)
    i=tl.load(REM+row*8+slot,slot<8,-1);j=tl.load(ADD+row*8+slot-8,slot>=8,-1)
    page=tl.where(slot<8,i,j);valid=page>=0
    start=tl.load(START+page,valid,0);count=tl.load(COUNT+page,valid,0)
    cost=tl.load(C+page,valid,1.)
    w=tl.load(W+((h*tr.cdiv(N,QT)+q//QT)*R+start[:,None]+p[None,:])*QT+q%QT,valid[:,None]&(p[None,:]<count[:,None]),0.).to(tl.float32)/cost[:,None]
    v=tl.load(V+(h*R+start[:,None,None]+p[None,:,None])*32+d[None,None,:],valid[:,None,None]&(p[None,:,None]<count[:,None,None]),0.)
    content=tl.sum(w[:,:,None]*v,1)
    tl.store(CV+(row*16+slot[:,None])*32+d[None,:],content)

