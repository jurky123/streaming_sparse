import ctypes,torch,triton
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.video_runtime_v1.fixed15_layout import pack_range
from local_experiments.high_sparse_runtime import _budget_mask
def rand(*shape):
    return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
lib=ctypes.CDLL('/tmp/tier_variable.so')
fn=lib.run
fn.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*2+[ctypes.c_void_p]
torch.manual_seed(701)
q=rand(1,4608,12,128)
for p,count in [(216,33),(432,43),(432,64),(432,105)]:
    k,v=[rand(1,p*64,12,128) for _ in range(2)]
    pk,pv=torch.empty_like(k),torch.empty_like(v)
    pack_range[(p,12,16)](k,v,pk,pv,0)
    keys,mean,detail,basis=rand(12,p,4,128),rand(12,p,128),rand(12,p,3,8),rand(12,128,8)
    scores=torch.rand(12,72,p,device='cuda')
    scores[...,:24]+=2
    own=torch.arange(p-72,p,device='cuda').expand(12,-1)
    scores.scatter_(-1,own[...,None],4.)
    ids=scores.argsort(-1,descending=True)[...,:count].int().contiguous()
    out=torch.empty_like(q)
    exact=torch.empty_like(q,dtype=torch.float)
    lse=torch.empty(12,4608,device='cuda')
    bits=torch.empty(12,72,16,device='cuda',dtype=torch.uint32)
    _budget_mask[(72,12)](ids,bits,72,count,triton.next_power_of_2(count))
    args=q,keys,mean,detail
    kw=dict(HEAD_PAGES=p,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,num_warps=4,num_stages=2,enable_fp_fusion=False)
    def baseline():
        _tail[(72,12)](*args,None,None,None,None,p,4608,12,128,K=k,V=v,IDS=ids,EXACT_COUNT=count,QK_SCALE=128**-.5,**kw)
    from local_experiments.video_runtime_v1.fixed15_layout import encode
    packed=torch.empty(12,(p+15)//16,10624,device='cuda',dtype=q.dtype)
    encode[(12,(p+15)//16,42)](keys,mean,detail,packed,p,p,(p+15)//16,DIRECT_MASS=True)
    flags=torch.empty(12,72,device='cuda',dtype=torch.int32)
    def candidate():
        assert not fn(*(x.data_ptr() for x in (q,pk,pv,ids,out,packed,basis,flags)),p,count,torch.cuda.current_stream().cuda_stream)
    baseline()
    ref=out.clone()
    candidate()
    diff=(out.float()-ref.float())
    print(p,count,flags.sum().item(),(diff!=0).float().mean().item(),diff.abs().max().item(),(diff.square().sum()/ref.float().square().sum()).item(),*[triton.testing.do_bench_cudagraph(f,rep=100) for f in (baseline,candidate)],flush=True)
