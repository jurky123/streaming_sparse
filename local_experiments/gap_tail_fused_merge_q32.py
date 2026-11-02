"""Candidate: SingleTail with rank-8 decode and exact merge in its epilogue.
May increase register pressure; not adopted without real-operand measurements.
"""
import math,torch,triton as tr,triton.language as tl
from .compensation_stable_kernels import _round_tf32
from .high_sparse_runtime import _budget_mask
@tr.jit
def _decode(BASE,DETAIL,OUT,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);d=i%8;r=i//8%4;p=i//32
    a=tl.load(BASE+p*8+d,i<N,0).to(tl.float32)
    x=tl.load(DETAIL+p*24+d,i<N,0).to(tl.float32)
    y=tl.load(DETAIL+p*24+8+d,i<N,0).to(tl.float32)
    z=tl.load(DETAIL+p*24+16+d,i<N,0).to(tl.float32)
    value=((a+tl.where(r<2,x,-x))+tl.where(r%2==0,y,-y))+tl.where((r==0)|(r==3),z,-z)
    tl.store(OUT+i,value,i<N)
@tr.jit
def _precision_tail(Q,HI,PV,MASK,OUT,LSE,SCALE,EXACT,ELSE,BASIS,CENTER,GAMMA,MODE:tl.constexpr,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64):
    tile=tl.program_id(0);h=tl.program_id(1);qi=tile*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);v=tl.arange(0,16)
    q=tl.load(Q+qi[:,None]*H*D+h*D+d[None,:],qi[:,None]<N,0).to(tl.float32)
    mi=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,16),tl.float32)
    for first in range(tr.cdiv(P*4,BR)):
        rep=first*BR+r;page=rep//4
        word=tl.load(MASK+(h*(N//64)+tile*BQ//64)*16+page//32)
        omitted=((word>>(page%32))&1)==0
        kh=tl.load(HI+(h*P*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
        logits=tl.dot(q.to(tl.bfloat16),kh)*(1.4426950408889634/math.sqrt(D))
        logits=tl.where((rep[None,:]<P*4)&omitted[None,:],logits,-float('inf'))
        new=tl.maximum(mi,tl.max(logits,1));fade=tl.exp2(mi-new);prob=tl.exp2(logits-new[:,None])
        values=tl.load(PV+(h*P*4+rep[:,None])*8+v[None,:],(rep[:,None]<P*4)&(v[None,:]<8),0)
        if MODE==1:
            ph=_round_tf32(prob);vh=_round_tf32(values)
            update=tl.dot(ph,vh,input_precision='tf32')
        elif MODE==2:
            update=tl.dot(prob.to(tl.bfloat16),values.to(tl.bfloat16))
        else:
            ph=prob.to(tl.bfloat16);pl=(prob-ph.to(tl.float32)).to(tl.bfloat16)
            vh=values.to(tl.bfloat16);vl=(values-vh.to(tl.float32)).to(tl.bfloat16)
            update=tl.dot(pl,vh)
            update=tl.dot(ph,vl,update)
            update=tl.dot(ph,vh,update)
        acc=acc*fade[:,None]+update;den=den*fade+tl.sum(prob,1);mi=new
    tail=acc/den[:,None]
    lt=mi+tl.log2(den)+4.+tl.load(SCALE+h)*1.4426950408889634
    # Retain diagnostic outputs for the same public API. Inference can later
    # elide these stores only after the fused variant is validated.
    tl.store(OUT+h*N*8+qi[:,None]*8+v[None,:],tail,(qi[:,None]<N)&(v[None,:]<8))
    tl.store(LSE+h*N+qi,lt,qi<N)
    basis=tl.load(BASIS+h*D*8+d[None,:]*8+v[:,None],v[:,None]<8,0).to(tl.float32)
    # Match the separate merge: padded rank lanes were loaded as zero.
    tail=tl.where(v[None,:]<8,tail,0.)
    value=tl.dot(tail,basis,input_precision='tf32x3')+tl.load(CENTER+h*D+d)[None,:]
    ls=tl.load(ELSE+h*N+qi,qi<N,0)
    rho=1/(1+tl.exp2(ls-lt));gamma=tl.load(GAMMA+h)
    original=tl.load(EXACT+qi[:,None]*H*D+h*D+d[None,:],qi[:,None]<N,0).to(tl.float32)
    result=original+(gamma*rho)[:,None]*(value-original)
    tl.store(EXACT+qi[:,None]*H*D+h*D+d[None,:],result,qi[:,None]<N)

def compensate(q,pk,base,detail,choice,lut,out,lse,basis,center,gamma,log_scale,mode=3):
    _,n,h,d=q.shape;p=pk.shape[1]
    hi=pk.flatten(1,2).to(torch.bfloat16)
    pv=torch.empty(h,p,4,8,device=q.device)
    _decode[(tr.cdiv(pv.numel(),256),)](base,detail,pv,pv.numel(),256,enable_fp_fusion=False)
    bits=torch.empty(h,n//64,16,device=q.device,dtype=torch.uint32)
    _budget_mask[(n//64,h)](lut,bits,n//64,lut.shape[-1],tr.next_power_of_2(lut.shape[-1]))
    tail=torch.empty(h,n,8,device=q.device);lt=torch.empty(h,n,device=q.device)
    _precision_tail[(n//32,h)](q,hi,pv,bits,tail,lt,log_scale,out,lse,basis,center,gamma,mode,p,n,h,d,32,64,num_warps=4,num_stages=2)
    return out,tail,lt
