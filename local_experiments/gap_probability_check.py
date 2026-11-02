import torch,triton,json
from .gap_probability_stats import probability_stats
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(92);rows=[]
    for p in [216,410]:
        prob=torch.randn(12,72,16,p,device='cuda').softmax(-1)
        def native():return prob.mean(2),prob.max(2).values,prob.clamp_min(1e-12).log().std(2,unbiased=False).clamp(0,5)/5
        ref=native()
        for fma in [True,False]:
            out=probability_stats(prob,fma=fma);row=dict(p=p,fma=fma,exact=[torch.equal(a,b) for a,b in zip(ref,out)],max_error=[float((a-b).abs().max()) for a,b in zip(ref,out)],native_ms=triton.testing.do_bench_cudagraph(native,rep=100),ms=triton.testing.do_bench_cudagraph(lambda:probability_stats(prob,fma=fma),rep=100));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_stats_20261031/probability_checks.json',dict(rows=rows))
if __name__=='__main__':main()
