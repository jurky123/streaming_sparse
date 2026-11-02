import torch,triton,json
from .gap_spatial_softmax import spatial
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(81);rows=[]
    for p,t in [(1,4608),(72,64)]:
        x=torch.randn(12,p,t,32,device='cuda');ref=x.softmax(-2)
        for b in [1,4,8]:
            for mul,exp in [(False,0),(True,0),(False,1)]:
                y=spatial(x,b,mul,exp);row=dict(p=p,t=t,block=b,mul=mul,exp=exp,exact=torch.equal(y,ref),max_error=float((y-ref).abs().max()),native_ms=triton.testing.do_bench_cudagraph(lambda:x.softmax(-2),rep=100),ms=triton.testing.do_bench_cudagraph(lambda:spatial(x,b,mul,exp),rep=100));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_stats_20261031/spatial_checks.json',dict(rows=rows))
if __name__=='__main__':main()
