"""Paired-V tail tiling sweep; same arithmetic, no algorithm or cache change."""
import json,torch,triton
from triton.runtime.errors import OutOfResources
from .conditional_v_kernels import _conditional_tail
from .high_sparse_runtime import _budget_mask
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(741);rows=[];n,h,d,p,keep=4608,12,128,410,41
    q=torch.randn(1,n,h,d,device='cuda',dtype=torch.bfloat16);pk=(torch.randn(h,p,4,d,device='cuda')*.2).contiguous()
    base=torch.randn(h,p,8,device='cuda',dtype=torch.bfloat16);detail=torch.randn(h,p,3,8,device='cuda',dtype=torch.bfloat16)*.1
    lut=torch.rand(h,n//64,p,device='cuda').topk(keep,-1).indices.sort(-1).values[None].int();bits=torch.empty(h,n//64,16,device='cuda',dtype=torch.uint32)
    _budget_mask[(n//64,h)](lut,bits,n//64,keep,triton.next_power_of_2(keep));scale=torch.zeros(h,device='cuda');reference=None
    for bq,bp,warps in [(64,16,4),(32,16,4),(64,32,4),(32,32,4),(64,32,8),(32,64,4),(64,64,8)]:
        out=torch.empty(h,n,8,device='cuda');lse=torch.empty(h,n,device='cuda')
        def run():_conditional_tail[(triton.cdiv(n,bq),h)](q,pk,base,detail,lut,bits,out,lse,scale,p,n,h,d,3,bq,bp,num_warps=warps,num_stages=1)
        try:
            run();torch.cuda.synchronize()
            if reference is None:reference=(out.clone(),lse.clone())
            nmse=float((out-reference[0]).square().sum()/reference[0].square().sum());le=float((lse-reference[1]).abs().max());assert nmse<1e-9 and le<2e-5,(nmse,le)
            ms=triton.testing.do_bench_cudagraph(run,rep=500);rows.append(dict(bq=bq,bp=bp,warps=warps,ms=ms,nmse=nmse,lse_max_error=le))
        except OutOfResources as exc:rows.append(dict(bq=bq,bp=bp,warps=warps,unavailable=str(exc)))
        atomic('artifacts/sparse_expansion_20261031/conditional_tuning.json',dict(rows=rows,status='running'));print(json.dumps(rows[-1]),flush=True)
    valid=[r for r in rows if 'ms' in r];best=min(valid,key=lambda r:r['ms'])
    atomic('artifacts/sparse_expansion_20261031/conditional_tuning.json',dict(status='complete',rows=rows,best=best,scope='synthetic steady-state paired-V tail only; full real-QKV/video validation required before deployment'))
if __name__=='__main__':main()
