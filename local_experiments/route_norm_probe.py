import torch,triton as tr,triton.language as tl,math
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def norm(R,O,N:tl.constexpr,ORDER:tl.constexpr,SCALE:tl.constexpr):
    x=tl.program_id(0)*256+tl.arange(0,256)
    a=tl.load(R+x*4+0,x<N,0);b=tl.load(R+x*4+1,x<N,0);c=tl.load(R+x*4+2,x<N,0);d=tl.load(R+x*4+3,x<N,0)
    if SCALE==0:a=a*(1./11.313708498984761);b=b*(1./11.313708498984761);c=c*(1./11.313708498984761);d=d*(1./11.313708498984761)
    else:a=tl.div_rn(a,11.313708498984761);b=tl.div_rn(b,11.313708498984761);c=tl.div_rn(c,11.313708498984761);d=tl.div_rn(d,11.313708498984761)
    mx=tl.maximum(tl.maximum(a,b),tl.maximum(c,d));a=lib.exp(a-mx);b=lib.exp(b-mx);c=lib.exp(c-mx);d=lib.exp(d-mx)
    if ORDER==0:s=(a+b)+(c+d)
    elif ORDER==1:s=(a+c)+(b+d)
    elif ORDER==2:s=((a+b)+c)+d
    else:s=((a+d)+b)+c
    tl.store(O+x,(lib.log(s)+mx)+2.772588722239781,x<N)
def main():
    torch.manual_seed(31);x=torch.randn(12,288,410,4,device='cuda')*10.;o=torch.empty_like(x[...,0]);ref=(x/math.sqrt(128)).logsumexp(-1)+math.log(16)
    for order in range(4):
        for scale in range(2):
            norm[(tr.cdiv(o.numel(),256),)](x,o,o.numel(),order,scale,enable_fp_fusion=False)
            print(order,scale,'neq',int((o!=ref).sum()),'max',float((o-ref).abs().max()),flush=True)
if __name__=='__main__':main()
