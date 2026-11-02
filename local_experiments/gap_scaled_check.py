import torch,triton,json,math
from .gap_scaled_stats import scaled_stats
from .gap_vector_stats import stats
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(171);rows=[]
    for p in [216,410]:
        for magnitude in [.1,10.,1000.]:
            x=torch.randn(12,72,16,p,4,device='cuda')*magnitude
            def native():return stats(x/math.sqrt(128))
            a=native();b=scaled_stats(x);row=dict(p=p,magnitude=magnitude,exact=[torch.equal(v,w) for v,w in zip(a,b)],errors=[float((v-w).abs().max()) for v,w in zip(a,b)],native_ms=triton.testing.do_bench_cudagraph(native,rep=100),ms=triton.testing.do_bench_cudagraph(lambda:scaled_stats(x),rep=100));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_memory_20261031/scaled_checks.json',dict(rows=rows))
if __name__=='__main__':main()
