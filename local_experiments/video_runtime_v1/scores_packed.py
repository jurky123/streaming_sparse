"""FP64 endpoint scoring directly reads packed candidates; no per-field copies."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@tr.jit
def packed_kernel(DEN,STATE,CAND,LB,RX,RO,AX,AO,MAX,OUT,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr,D:tl.constexpr,ACTION=None,LAZY:tl.constexpr=False):
    action=tl.program_id(0) if ACTION is None else ACTION
    batch=action//(R*A);ri=(action//A)%R;ai=action%A
    m=tl.arange(0,M);d=tl.arange(0,D)
    den=tl.load(DEN+((batch*M+m)*R+ri)*A+ai)
    hr=tl.load(STATE+(batch*M+m[:,None])*3*D+d[None,:]);err=tl.load(STATE+(batch*M+m[:,None])*3*D+D+d[None,:])
    rpbase=CAND+((batch*M+m)*(R+A)+ri)*34
    apbase=CAND+((batch*M+m)*(R+A)+R+ai)*34
    lr=tl.load(rpbase);br=tl.load(rpbase+1);er=lr+br
    la=tl.load(apbase);ba=tl.load(apbase+1);ea=la+ba
    cr=tl.load(rpbase[:,None]+2+d[None,:]);pr=tl.load(rpbase[:,None]+18+d[None,:])
    ca=tl.load(apbase[:,None]+2+d[None,:]);pa=tl.load(apbase[:,None]+18+d[None,:])
    xer=lib.exp(er-den);xar=lib.exp(lr-den);xea=lib.exp(ea-den);xaa=lib.exp(la-den)
    dr=tl.where(tl.abs(br)<.5,xar*lib.expm1(tl.minimum(tl.maximum(br,-.5),.5)),xer-xar)
    da=tl.where(tl.abs(ba)<.5,xaa*lib.expm1(tl.minimum(tl.maximum(ba,-.5),.5)),xea-xaa)
    delta=da[:,None]*(ca-hr)+xea[:,None]*pa-(dr[:,None]*(cr-hr)+xer[:,None]*pr)
    maximum=tl.load(MAX+batch*M+m)
    finite=(tl.abs(delta)<float('inf'))
    bad=(maximum-den>80)|(tl.sum(finite.to(tl.int32),1)!=D)
    if tl.sum(bad.to(tl.int32),0)>0:
        lb=tl.load(LB+batch*M+m);rx=tl.load(RX+(batch*M+m)*R+ri);ax=tl.load(AX+(batch*M+m)*A+ai)
        ob=tl.load(STATE+(batch*M+m[:,None])*3*D+2*D+d[None,:])
        if LAZY:
            # Reconstruct only for unstable actions, summing positive page masses.
            new=lib.exp(lb-den)[:,None]*ob
            for j in range(R):
                if j!=ri:
                    ptr=CAND+((batch*M+m)*(R+A)+j)*34
                    ll=tl.load(ptr)+tl.load(ptr+1)
                    cc=tl.load(ptr[:,None]+2+d[None,:]);pp=tl.load(ptr[:,None]+18+d[None,:])
                    new=new+lib.exp(ll-den)[:,None]*(cc+pp)
            for j in range(A):
                if j!=ai:
                    ptr=CAND+((batch*M+m)*(R+A)+R+j)*34
                    ll=tl.load(ptr);cc=tl.load(ptr[:,None]+2+d[None,:])
                    new=new+lib.exp(ll-den)[:,None]*cc
        else:
            ro=tl.load(RO+((batch*M+m[:,None])*R+ri)*D+d[None,:]);ao=tl.load(AO+((batch*M+m[:,None])*A+ai)*D+d[None,:])
            new=lib.exp(lb-den)[:,None]*ob+lib.exp(rx-den)[:,None]*ro
            new=new+lib.exp(ax-den)[:,None]*ao
        new=new+xar[:,None]*cr
        new=new+xea[:,None]*(ca+pa)
        delta=tl.where(bad[:,None],new-hr,delta)
    gain=2*tl.sum(err*delta,1)-tl.sum(delta*delta,1)
    tl.store(OUT+action,tl.sum(gain,0)/M)

@torch.library.custom_op('streaming_sparse::packed_reduce',mutates_args=())
def packed_reduce(den:torch.Tensor,state:torch.Tensor,cand:torch.Tensor,lb:torch.Tensor,rx:torch.Tensor,ro:torch.Tensor,ax:torch.Tensor,ao:torch.Tensor,maximum:torch.Tensor)->torch.Tensor:
    b,m,r,a=den.shape;d=state.shape[-1];out=torch.empty((b,r,a),device=den.device,dtype=torch.float64)
    args=[x.contiguous() for x in (den,state,cand,lb,rx,ro,ax,ao,maximum)]
    packed_kernel[(b*r*a,)](*args,out,r,a,m,d,num_warps=4,enable_fp_fusion=False)
    return out

@packed_reduce.register_fake
def _(den,state,cand,lb,rx,ro,ax,ao,maximum):
    return den.new_empty((den.shape[0],den.shape[2],den.shape[3]))
