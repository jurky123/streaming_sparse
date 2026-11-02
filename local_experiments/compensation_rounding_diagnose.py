"""Locate first floating-point divergence between query tile geometries."""
import math,json
from pathlib import Path
import torch,triton as tr,triton.language as tl
from .learned_cache.io import atomic
from .compensation_fast_kernels import mask,tail
@tr.jit
def stages(Q,K,L,Z,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,BQ:tl.constexpr):
    qi=tl.program_id(0)*BQ+tl.arange(0,BQ);h=tl.program_id(1);di=tl.arange(0,D);ri=tl.arange(0,64)
    q=tl.load(Q+qi[:,None]*H*D+h*D+di[None,:],qi[:,None]<N,0).to(tl.float32)
    k=tl.load(K+h*CAP*4*D+ri[None,:]*D+di[:,None])
    log=tl.dot(q,k,input_precision='tf32x3')*(1.4426950408889634/math.sqrt(D))
    tl.store(L+(h*N+qi[:,None])*64+ri[None,:],log,qi[:,None]<N)
    group=tl.reshape(log,(BQ,16,4));mx=tl.max(group,2)
    z=mx+tl.log2(tl.sum(tl.exp2(group-mx[:,:,None]),2))+4.
    tl.store(Z+(h*N+qi[:,None])*16+tl.arange(0,16)[None,:],z,qi[:,None]<N)
def diff(a,b):
    d=(a-b).abs();return dict(equal=torch.equal(a,b),count=int((a!=b).sum()),max=float(d.max()),total=a.numel())
@torch.no_grad()
def main():
    d={k:v.cuda() for k,v in torch.load('artifacts/compensation_kernel_opt_20261030/input.pt',weights_only=True).items()};q=d['q'];_,n,h,dim=q.shape
    bits=mask(d['lut'],n);results=[];baseline=None
    for bq in [16,32,64]:
        log=torch.empty(h,n,64,device='cuda');z=torch.empty(h,n,16,device='cuda')
        stages[(tr.cdiv(n,bq),h)](q,d['pk'],log,z,n,h,dim,338,bq,num_warps=4)
        ta,ls,_=tail(q,d['pk'],d['code'],bits,338,(bq,16,4,1,False))
        if baseline is None:baseline=[log,z,ta,ls]
        row=dict(bq=bq,**{name:diff(a,b) for name,a,b in zip(['logits','page_z','tail','lse'],[log,z,ta,ls],baseline)})
        results.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/compensation_kernel_opt_20261030/rounding_diagnosis.json',results)
if __name__=='__main__':main()
