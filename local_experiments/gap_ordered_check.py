import torch,triton,json
from .gap_repstats import stats as reference
from .gap_vector_stats import stats
from .gap_weight_ordered import weight
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    from .gap_group_isolated import device_peers
    assert not device_peers(),"GPU has peers"
    torch.set_num_threads(4);torch.manual_seed(712);rows=[]
    for p in [216,288,360,410]:
        x=torch.randn(12,72,16,p,4,device='cuda')*4
        z,w=reference(x,div=1);prob=z.softmax(-1);mass=prob.mean(2)
        refweight=(prob[...,None]*w).mean(2)/mass[...,None].clamp_min(1e-12)
        for b in [8,16,32,64]:
            y=weight(prob,w,mass,b);row=dict(kind='weight',p=p,block=b,exact=torch.equal(y,refweight),maxerr=float((y-refweight).abs().max()))
            if p==410:row['ms']=triton.testing.do_bench_cudagraph(lambda:weight(prob,w,mass,b),rep=200)
            rows.append(row);print(json.dumps(row),flush=True)
        for b in [128,256,512,1024]:
            a,c=stats(x,block=b);row=dict(kind='stats',p=p,block=b,exact=torch.equal(a,z) and torch.equal(c,w))
            if p==410:row['ms']=triton.testing.do_bench_cudagraph(lambda:stats(x,block=b),rep=200)
            rows.append(row);print(json.dumps(row),flush=True)
        if p==410:
            rows.append(dict(kind='reference_stats',ms=triton.testing.do_bench_cudagraph(lambda:reference(x,div=1),rep=200)))
            rows.append(dict(kind='reference_weight',ms=triton.testing.do_bench_cudagraph(lambda:(prob[...,None]*w).mean(2)/mass[...,None].clamp_min(1e-12),rep=200)))
    atomic('artifacts/gap_vector_flat_20261031/ordered_weight_checks.json',dict(rows=rows))
if __name__=='__main__':main()
