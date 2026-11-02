import torch,triton,json
from .gap_gather_runtime import balanced_gather as balanced
from .gap_tensor_gram import balanced_tensor_gram as balanced_gather
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(639);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    for h,n,factors in [(12,4608,[3,3,2,2,2]),(864,64,[2,2])]:
        for scale in [1.e-4,1.,100.]:
            x=torch.randn(h,n,32,device='cuda')*scale
            ref=balanced(x,factors);out=balanced_gather(x,factors)
            row=dict(h=h,n=n,scale=scale,exact=torch.equal(ref,out),permutation_agreement=float((ref==out).float().mean()),native_ms=triton.testing.do_bench_cudagraph(lambda:balanced(x,factors),rep=100),ms=triton.testing.do_bench_cudagraph(lambda:balanced_gather(x,factors),rep=100));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_tensor_gram_20261101/synthetic_checks.json',dict(rows=rows))
if __name__=='__main__':main()
