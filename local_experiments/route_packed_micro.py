import json,time
from pathlib import Path
import torch
from .route_graph_runtime import RouteGraph
from .route_packed_runtime import PackedRouter as FusedRouter
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(31);torch.backends.cuda.matmul.allow_tf32=False
    old=RouteGraph();new=FusedRouter();rows=[]
    for p in [216,288,360,410]:
        q=torch.randn(1,4608,12,128,device='cuda',dtype=torch.bfloat16);pk=torch.randn(12,p,4,128,device='cuda');c=torch.randn(12,p,8,device='cuda',dtype=torch.bfloat16);v=torch.rand(12,p,device='cuda');w=torch.randn(12,14,device='cuda')*.3
        a=old(q,pk,c,v,w,True)[0].clone();b=new(q,pk,c,v,w,True)[0];diff=int((a!=b).any(-1).sum());print('difference',p,diff,flush=True)
        for label,r in [('graph',old),('fused',new)]:
            for _ in range(5):r(q,pk,c,v,w,True)
            for rep in range(3):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();t=time.perf_counter()
                for _ in range(50):r(q,pk,c,v,w,True)
                b.record();b.synchronize();rows.append(dict(p=p,method=label,rep=rep,ms=a.elapsed_time(b)/50,different_tiles=diff))
        print(rows[-6:],flush=True)
    atomic(Path('artifacts/route_kernel_20261031/packed_micro.json'),rows)
if __name__=='__main__':main()
