"""Inference-only fused page features/score and one-swap LUT. FP32, no TF32."""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
import torch
@tr.jit
def _score(PROB,MASS,CODE,VV,KV,W,MASK,DEN,SEL,ALL,S,P:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);h=row//72;t=row%72;p=tl.arange(0,B);d=tl.arange(0,8)
    mass=tl.load(MASS+row*P+p,p<P,0);lm=lib.log(tl.maximum(mass,1.e-12));lc=tl.minimum(tl.maximum(lm,-20.),0.)/5.
    a=tl.load(PROB+((h*72+t)*4+0)*P+p,p<P,0);b=tl.load(PROB+((h*72+t)*4+1)*P+p,p<P,0)
    c=tl.load(PROB+((h*72+t)*4+2)*P+p,p<P,0);e=tl.load(PROB+((h*72+t)*4+3)*P+p,p<P,0)
    la=lib.log(tl.maximum(a,1.e-12));lb=lib.log(tl.maximum(b,1.e-12));lc2=lib.log(tl.maximum(c,1.e-12));le=lib.log(tl.maximum(e,1.e-12))
    avg=(la+lb+lc2+le)*.25
    spread=tl.minimum(tl.sqrt(((la-avg)*(la-avg)+(lb-avg)*(lb-avg)+(lc2-avg)*(lc2-avg)+(le-avg)*(le-avg))*.25),5.)/5.
    peak=tl.minimum(tl.maximum(lib.log(tl.maximum(tl.maximum(tl.maximum(a,b),tl.maximum(c,e)),1.e-12))-lm,0.),10.)/5.
    vv=lib.log1p(tl.maximum(tl.load(VV+h*P+p,p<P,0),0.));kv=tl.load(KV+h*P+p,p<P,0)
    current=(p>=P-72).to(tl.float32);mask=tl.load(MASK+row*P+p,p<P,0).to(tl.float32);den=tl.load(DEN+row)
    sel=tl.load(SEL+row*8+d);missing=tl.load(ALL+row*8+d)-sel
    code=tl.load(CODE+(h*P+p[:,None])*8+d[None,:],p[:,None]<P,0).to(tl.float32);diff=code-sel[None,:]
    energy=tl.maximum(tl.sum(sel*sel,0),.01);align=tl.sum(diff*missing[None,:],1)/energy;mag=tl.sum(diff*diff,1)/energy
    dot=tl.load(W+h*14+0)+lc*tl.load(W+h*14+1)+peak*tl.load(W+h*14+2)+spread*tl.load(W+h*14+3)
    dot=dot+vv*tl.load(W+h*14+4)+kv*tl.load(W+h*14+5)+current*tl.load(W+h*14+6)+(current*lc)*tl.load(W+h*14+7)
    dot=dot+mask*tl.load(W+h*14+8)+tl.minimum(mass/den,2.)*tl.load(W+h*14+9)
    dot=dot+(tl.minimum(tl.maximum(align,-10.),10.)/5.)*tl.load(W+h*14+10)+(tl.minimum(mag,10.)/5.)*tl.load(W+h*14+11)
    dot=dot+tl.minimum(tl.maximum(mass*align,-2.),2.)*tl.load(W+h*14+12)+tl.minimum(mass*mag,2.)*tl.load(W+h*14+13)
    tl.store(S+row*P+p,lm+lib.tanh(dot),p<P)
@tr.jit
def _lut(S,MASK,IDS,OUT,P:tl.constexpr,KEEP:tl.constexpr,B:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,B);i=tl.arange(0,K);own=row%72+P-72
    s=tl.load(S+row*P+p,p<P,0);mask=tl.load(MASK+row*P+p,p<P,0)
    low=tl.min(tl.where((p<P)&mask&(p!=own),s,float('inf')),0);high=tl.max(tl.where((p<P)&(~mask),s,-float('inf')),0)
    drop=tl.min(tl.where((p<P)&mask&(p!=own)&(s==low),p,2147483647),0);add=tl.min(tl.where((p<P)&(~mask)&(s==high),p,2147483647),0)
    ids=tl.load(IDS+row*KEEP+i,i<KEEP,2147483647).to(tl.int32);ids=tl.where((ids==drop)&(high>low),add,ids)
    ids=tl.sort(ids,descending=False);tl.store(OUT+row*KEEP+i,ids,i<KEEP)
def score_lut(prob,mass,code,vv,kv,w,mask,den,sel,allvalue,ids):
    p=mass.shape[-1];keep=ids.shape[-1];s=torch.empty_like(mass);out=torch.empty((1,12,72,keep),device=mass.device,dtype=torch.int32)
    _score[(864,)](prob,mass,code,vv,kv,w,mask,den,sel,allvalue,s,p,tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False)
    _lut[(864,)](s,mask,ids,out,p,keep,tr.next_power_of_2(p),tr.next_power_of_2(keep),num_warps=4)
    return out

@tr.jit
def _summary(K,V,PK,VM,VAR,OFFSET:tl.constexpr):
    page=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,128)
    off=((OFFSET+page*64+t[:,None])*12+h)*128+d[None,:]
    v=tl.load(V+off).to(tl.float32);mean=tl.sum(v,0)/64
    var=tl.maximum(tl.sum(tl.sum(v*v,0)/64-mean*mean,0)/128,0.)
    k=tl.load(K+off).to(tl.float32);kp=tl.sum(tl.reshape(k,(4,16,128)),1)/16
    tl.store(PK+((h*72+page)*4+tl.arange(0,4)[:,None])*128+d[None,:],kp)
    tl.store(VM+(h*72+page)*128+d,mean);tl.store(VAR+h*72+page,var)
def summary(cache,offset):
    pk=torch.empty(12,72,4,128,device='cuda');vm=torch.empty(12,72,128,device='cuda');var=torch.empty(12,72,device='cuda',dtype=torch.bfloat16)
    _summary[(72,12)](cache['k'],cache['v'],pk,vm,var,offset,num_warps=4)
    return pk,vm,var

@tr.jit
def _gather(Q,OUT,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B);d=x%128;probe=x//128%4;tile=x//512%72;h=x//(512*72)
    # Same four probes and layout as native index_select -> permute -> float.
    src=(((tile*64+probe*21)*12+h)*128+d)
    tl.store(OUT+x,tl.load(Q+src,x<12*288*128,0).to(tl.float32),x<12*288*128)
def gather(q,out):_gather[(tr.cdiv(out.numel(),1024),)](q,out,1024)
