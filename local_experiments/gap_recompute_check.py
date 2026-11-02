import torch,triton,json
from .gap_weight_recompute import logz,weight
from .gap_vector_stats import stats
from .gap_weight_ordered import weight as native_weight
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(730);rows=[]
    for p in [216,410]:
        x=torch.randn(12,72,16,p,4,device='cuda')*3
        z,w=stats(x);prob=z.softmax(-1);mass=prob.mean(2);ref=native_weight(prob,w,mass,block=64)
        def native():
            z,w=stats(x);return native_weight(prob,w,mass,block=64)
        for b in [16,32,64,128]:
            def fused():z=logz(x);return weight(prob,x,mass,b)
            out=fused();row=dict(p=p,block=b,logz_exact=torch.equal(logz(x),z),weight_exact=torch.equal(out,ref),max_error=float((out-ref).abs().max()),native_ms=triton.testing.do_bench_cudagraph(native,rep=100),ms=triton.testing.do_bench_cudagraph(fused,rep=100));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_memory_20261031/recompute_checks.json',dict(rows=rows))
if __name__=='__main__':main()
