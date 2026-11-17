"""Factor common action normalization; FP64 endpoint coefficients shared by actions."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@tr.jit
def factored_kernel(DEN,H,E,LR,ER,BR,CR,PR,LA,EA,BA,CA,PA,LB,OB,RX,RO,AX,AO,MAX,XR,DR,XA,DA,OUT,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr,D:tl.constexpr):
    action=tl.program_id(0);batch=action//(R*A);ri=(action//A)%R;ai=action%A
    m=tl.arange(0,M);d=tl.arange(0,D)
    den=tl.load(DEN+((batch*M+m)*R+ri)*A+ai)
    hr=tl.load(H+(batch*M+m[:,None])*D+d[None,:]);err=tl.load(E+(batch*M+m[:,None])*D+d[None,:])
    lr=tl.load(LR+(batch*M+m)*R+ri);er=tl.load(ER+(batch*M+m)*R+ri);br=tl.load(BR+(batch*M+m)*R+ri)
    la=tl.load(LA+(batch*M+m)*A+ai);ea=tl.load(EA+(batch*M+m)*A+ai);ba=tl.load(BA+(batch*M+m)*A+ai)
    cr=tl.load(CR+((batch*M+m[:,None])*R+ri)*D+d[None,:]);pr=tl.load(PR+((batch*M+m[:,None])*R+ri)*D+d[None,:])
    ca=tl.load(CA+((batch*M+m[:,None])*A+ai)*D+d[None,:]);pa=tl.load(PA+((batch*M+m[:,None])*A+ai)*D+d[None,:])
    maximum=tl.load(MAX+batch*M+m)
    scale=lib.exp(maximum-den)
    xer=tl.load(XR+(batch*M+m)*R+ri);dr=tl.load(DR+(batch*M+m)*R+ri)
    xea=tl.load(XA+(batch*M+m)*A+ai);da=tl.load(DA+(batch*M+m)*A+ai)
    delta=scale[:,None]*(da[:,None]*(ca-hr)+xea[:,None]*pa-(dr[:,None]*(cr-hr)+xer[:,None]*pr))
    finite=(tl.abs(delta)<float('inf'))
    bad=(maximum-den>80)|(tl.sum(finite.to(tl.int32),1)!=D)
    if tl.sum(bad.to(tl.int32),0)>0:
        lb=tl.load(LB+batch*M+m);rx=tl.load(RX+(batch*M+m)*R+ri);ax=tl.load(AX+(batch*M+m)*A+ai)
        ob=tl.load(OB+(batch*M+m[:,None])*D+d[None,:]);ro=tl.load(RO+((batch*M+m[:,None])*R+ri)*D+d[None,:]);ao=tl.load(AO+((batch*M+m[:,None])*A+ai)*D+d[None,:])
        new=lib.exp(lb-den)[:,None]*ob+lib.exp(rx-den)[:,None]*ro
        new=new+lib.exp(ax-den)[:,None]*ao
        new=new+lib.exp(lr-den)[:,None]*cr
        new=new+lib.exp(ea-den)[:,None]*(ca+pa)
        delta=tl.where(bad[:,None],new-hr,delta)
    gain=2*tl.sum(err*delta,1)-tl.sum(delta*delta,1)
    tl.store(OUT+action,tl.sum(gain,0)/M)

@torch.library.custom_op('streaming_sparse::factored_reduce',mutates_args=())
def factored_reduce(den:torch.Tensor,h:torch.Tensor,e:torch.Tensor,lr:torch.Tensor,er:torch.Tensor,br:torch.Tensor,cr:torch.Tensor,pr:torch.Tensor,la:torch.Tensor,ea:torch.Tensor,ba:torch.Tensor,ca:torch.Tensor,pa:torch.Tensor,lb:torch.Tensor,ob:torch.Tensor,rx:torch.Tensor,ro:torch.Tensor,ax:torch.Tensor,ao:torch.Tensor,maximum:torch.Tensor,xr:torch.Tensor,dr:torch.Tensor,xa:torch.Tensor,da:torch.Tensor)->torch.Tensor:
    b,m,r,a=den.shape;d=h.shape[-1];out=torch.empty((b,r,a),device=den.device,dtype=torch.float64)
    args=[x.contiguous() for x in (den,h,e,lr,er,br,cr,pr,la,ea,ba,ca,pa,lb,ob,rx,ro,ax,ao,maximum,xr,dr,xa,da)]
    factored_kernel[(b*r*a,)](*args,out,r,a,m,d,num_warps=4,enable_fp_fusion=False)
    return out

@factored_reduce.register_fake
def _(den,h,e,lr,er,br,cr,pr,la,ea,ba,ca,pa,lb,ob,rx,ro,ax,ao,maximum,xr,dr,xa,da):
    return den.new_empty((den.shape[0],den.shape[2],den.shape[3]))

def reduce(den,h,e,lr,er,br,cr,pr,la,ea,ba,ca,pa,lb,ob,rx,ro,ax,ao,maximum):
    def coefficients(l,ex,beta):
        approx=(l-maximum[...,None]).exp();exact=(ex-maximum[...,None]).exp()
        difference=torch.where(beta.abs()<.5,approx*torch.expm1(beta.clamp(-.5,.5)),exact-approx)
        return exact,difference
    xr,dr=coefficients(lr,er,br);xa,da=coefficients(la,ea,ba)
    return factored_reduce(den,h,e,lr,er,br,cr,pr,la,ea,ba,ca,pa,lb,ob,rx,ro,ax,ao,maximum,xr,dr,xa,da)
