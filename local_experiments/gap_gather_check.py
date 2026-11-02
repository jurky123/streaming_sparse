import torch,triton,json
from .gap_group_runtime import balanced
from .gap_gather_runtime import balanced_gather
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(639);rows=[]
    for h,n,factors in [(12,4608,[3,3,2,2,2]),(864,64,[2,2])]:
        for scale in [1.e-4,1.,100.]:
            x=torch.randn(h,n,32,device='cuda')*scale
            ref=balanced(x,factors);out=balanced_gather(x,factors)
            row=dict(h=h,n=n,scale=scale,exact=torch.equal(ref,out),native_ms=triton.testing.do_bench_cudagraph(lambda:balanced(x,factors),rep=100),ms=triton.testing.do_bench_cudagraph(lambda:balanced_gather(x,factors),rep=100));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_memory_20261031/gather_checks.json',dict(rows=rows))
if __name__=='__main__':main()
