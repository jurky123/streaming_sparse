"""Identify the accumulation orders used for the small-node FP32 cuBLAS GEMVs."""
import json
import torch
import triton as tr
import triton.language as tl
from .weights import ROOT

@tr.jit
def _reduce_terms(X,V,OUT,N:tl.constexpr,D:tl.constexpr,TRANS:tl.constexpr,
                  LANES:tl.constexpr,ASCENDING:tl.constexpr,CONTIG:tl.constexpr,FUSED:tl.constexpr,BITS:tl.constexpr):
    row=tl.program_id(0)
    group=tl.program_id(1)
    lane=tl.arange(0,LANES)
    length:tl.constexpr=N if TRANS else D
    steps:tl.constexpr=length//LANES
    prod=tl.full((LANES,),0,tl.float32)
    for i in range(steps):
        at=lane*steps+i if CONTIG else lane+i*LANES
        x=tl.load(X+group*N*D+(at*D+row if TRANS else row*D+at))
        v=tl.load(V+group*length+at)
        if FUSED:prod=tl.fma(x,v,prod)
        else:prod+=x*v
    for i in tl.static_range(BITS):
        distance=(1<<i) if ASCENDING else (LANES>>(i+1))
        peer=tl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
                                      constraints="=f,f,r",args=[prod,distance],dtype=tl.float32,is_pure=True,pack=1)
        prod+=peer
    tl.store(OUT+group*(D if TRANS else N)+row+lane,prod,lane==0)

@torch.no_grad()
def main():
    torch.manual_seed(701);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for n,groups in ((32,1728),(64,864),(128,432),(256,216),(512,108)):
        x=torch.randn(groups,n,32,device='cuda')
        for trans in (False,True):
            mat=x.transpose(1,2) if trans else x
            v=torch.randn(groups,mat.shape[-1],1,device='cuda')
            ref=torch.bmm(mat,v).squeeze(-1)
            out=torch.empty_like(ref)
            results=[]
            for lanes in (1,2,4,8,16,32):
                for asc in ((False,True) if lanes>2 else (False,)):
                    for contig in ((False,True) if 1<lanes<mat.shape[-1] else (False,)):
                        for fused in (False,True):
                            _reduce_terms[(mat.shape[1],groups)](x,v,out,n,32,trans,lanes,asc,contig,fused,lanes.bit_length()-1,num_warps=1,enable_fp_fusion=False)
                            same=(ref==out).float().mean().item()
                            results.append(dict(lanes=lanes,ascending=asc,contiguous=contig,fused=fused,same=same))
            result=dict(n=n,groups=groups,transpose=trans,orders=sorted(results,key=lambda r:-r['same'])[:5])
            print(n,trans,result['orders'][0],flush=True);rows.append(result)
    path=ROOT/'artifacts/video_runtime_v1/group_gemv_order_v1';path.mkdir(exist_ok=True)
    (path/'orders.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
