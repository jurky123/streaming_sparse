"""Paired-V tail tiling sweep; same arithmetic, no algorithm or cache change."""
import json,torch,triton
from triton.runtime.errors import OutOfResources
from .conditional_v_kernels import _conditional_tail
from .gap_reload_tail import _reload_tail
from .high_sparse_runtime import _budget_mask
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(741);rows=[];n,h,d,p,keep=4608,12,128,410,41
    q=torch.randn(1,n,h,d,device='cuda',dtype=torch.bfloat16);pk=(torch.randn(h,p,4,d,device='cuda')*.2).contiguous()
    base=torch.randn(h,p,8,device='cuda',dtype=torch.bfloat16);detail=torch.randn(h,p,3,8,device='cuda',dtype=torch.bfloat16)*.1
    lut=torch.rand(h,n//64,p,device='cuda').topk(keep,-1).indices.sort(-1).values[None].int();bits=torch.empty(h,n//64,16,device='cuda',dtype=torch.uint32)
    _budget_mask[(n//64,h)](lut,bits,n//64,keep,triton.next_power_of_2(keep));scale=torch.zeros(h,device='cuda');reference=None
    out=torch.empty(h,n,8,device='cuda');lse=torch.empty(h,n,device='cuda')
    def old():return _conditional_tail[(n//64,h)](q,pk,base,detail,lut,bits,out,lse,scale,p,n,h,d,3,64,16,num_warps=4,num_stages=1)
    old();target=(out.clone(),lse.clone())
    def kernel():return _reload_tail[(n//64,h)](q,pk,base,detail,lut,bits,out,lse,scale,p,n,h,d,3,64,16,num_warps=4,num_stages=1)
    compiled=kernel();exact=torch.equal(out,target[0]) and torch.equal(lse,target[1])
    row=dict(exact=exact,old_ms=triton.testing.do_bench_cudagraph(old,rep=500),tail_ms=triton.testing.do_bench_cudagraph(kernel,rep=500),regs=compiled.n_regs,shared=compiled.metadata.shared)
    print(json.dumps(row),flush=True);atomic('artifacts/gap_hotspots_20261031/reload.json',row)
if __name__=='__main__':main()
