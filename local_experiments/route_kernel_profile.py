import json,time
from pathlib import Path
import torch
from .route_graph_runtime import RouteGraph,postprocess
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    r=RouteGraph();q=torch.randn(1,4608,12,128,device='cuda',dtype=torch.bfloat16);pk=torch.randn(12,410,4,128,device='cuda');c=torch.randn(12,410,8,device='cuda',dtype=torch.bfloat16);v=torch.ones(12,410,device='cuda');w=torch.randn(12,14,device='cuda')
    for _ in range(10):r(q,pk,c,v,w,True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):r(q,pk,c,v,w,True)
        torch.cuda.synchronize()
    out=Path('artifacts/route_kernel_20261031');(out/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=45));print((out/'profile.txt').read_text())
    rows=[]
    for name,fn in [('route_graph_call',lambda:r(q,pk,c,v,w,True)),('post_graph',r.graphs[(410,True)].replay)]:
        for rep in range(5):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();t=time.perf_counter()
            for _ in range(50):fn()
            b.record();b.synchronize();rows.append(dict(name=name,rep=rep,gpu_ms=a.elapsed_time(b)/50,wall_ms=(time.perf_counter()-t)*20))
    atomic(out/'profile.json',rows);print(json.dumps(rows))
if __name__=='__main__':main()
