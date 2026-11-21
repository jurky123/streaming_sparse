import ctypes as c,torch,triton
from local_experiments.video_runtime_v1.ffn_rounded_gelu import library,gelu_table
P=c.c_void_p
lib=c.CDLL('/tmp/tier_lt.so')
lib.init_plan.argtypes=[c.c_int]*3+[P,c.c_int,c.POINTER(c.c_int)]
lib.init_plan.restype=P
lib.apply_plan.argtypes=[P,c.c_int]+[P]*5
m=4608
torch.manual_seed(701)
work=torch.empty(32<<20,device='cuda',dtype=torch.uint8)
stream=torch.cuda.current_stream().cuda_stream
for k,n,gelu in [(1536,1536,0),(1536,8960,1),(8960,1536,0)]:
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=x.dtype)/k**.5
    bias=torch.randn(n,device='cuda',dtype=x.dtype)
    y=x.new_empty(m,n)
    def baseline():
        if gelu:
            library().ffn_gelu(128,m,n,k,x.data_ptr(),w.data_ptr(),bias.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
        else:torch.addmm(bias,x,w.T,out=y)
    if gelu:gelu_table(x.device)
    baseline()
    ref=y.clone()
    num=c.c_int()
    plan=lib.init_plan(m,n,k,bias.data_ptr(),gelu,c.byref(num))
    results=[]
    for i in range(num.value):
        def run():return lib.apply_plan(plan,i,x.data_ptr(),w.data_ptr(),y.data_ptr(),work.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if run(): continue
        results.append((triton.testing.do_bench_cudagraph(run,rep=20),i))
    print(k,n,'baseline',triton.testing.do_bench_cudagraph(baseline,rep=100),'candidates',sorted(results)[:4],flush=True)
