import torch,triton,json
from .gap_graph_runtime import postprocess as ref
from .gap_fused_runtime import postprocess as stats
from .gap_feature_runtime import postprocess as features
from .conditional_v_kernels import HAAR
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(731);rows=[]
    for p in [216,288,360,410]:
        def rand(*shape):return torch.randn(*shape,device='cuda')
        for scale in [.2,1.,4.]:
            args=(rand(12,72,16,p,4)*scale,rand(12,72,16,8),rand(12,p,8).bfloat16(),rand(12,p,3,8).bfloat16(),rand(12,p,8).bfloat16(),rand(12,p).abs(),rand(12,p).abs(),rand(12,14)*.1,rand(12)*.1,torch.tensor(HAAR,device='cuda',dtype=torch.float32))
            expected=ref(*args)[0]
            row=dict(p=p,scale=scale)
            for name,fn in [('stats',stats),('features',features)]:
                value=fn(*args)[0];row[name+'_exact']=torch.equal(value,expected)
            if scale==1.:
                for name,fn in [('ref',ref),('stats',stats),('features',features)]:row[name+'_ms']=triton.testing.do_bench_cudagraph(lambda:fn(*args),rep=200)
            rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_profile_20261031/fusion_checks.json',dict(rows=rows,passed=all(r['stats_exact'] and r['features_exact'] for r in rows)))
if __name__=='__main__':main()
