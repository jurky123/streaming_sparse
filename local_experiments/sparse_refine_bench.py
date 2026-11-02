"""Matched CUDA-graph benchmark of native versus fused shortlist corrections."""
import json,torch,triton
from .sparse_expansion_runtime import refine as reference
from .sparse_expansion_fast_runtime import refine
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(739);rows=[]
    for p in [216,288,410]:
        mass=torch.randn(12,72,p,device='cuda').softmax(-1);feat=torch.randn(12,72,p,8,device='cuda');code=torch.randn(12,p,8,device='cuda',dtype=torch.bfloat16);w=torch.randn(12,14,device='cuda')*.1;own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1)
        for steps in [1,2,4]:
            args=(mass,feat,code,w,own,int(p*.1),steps)
            a,_=reference(*args);b,_=refine(*args);assert torch.equal(a,b)
            old=triton.testing.do_bench_cudagraph(lambda:reference(*args),rep=500)
            new=triton.testing.do_bench_cudagraph(lambda:refine(*args),rep=500)
            rows.append(dict(pages=p,steps=steps,reference_ms=old,fused_ms=new,speedup=old/new))
    result=dict(rows=rows,scope='synthetic fixed inputs, selection and refinement only; excludes probe QK, summaries, attention and end-to-end generation')
    atomic('artifacts/sparse_expansion_20261031/refine_benchmark.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
