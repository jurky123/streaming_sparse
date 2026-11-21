import torch,triton
from local_experiments.video_runtime_v1.reader_layout import reader_kernel
from local_experiments.video_runtime_v1.reader_split import scalar_kernel,vector_kernel
torch.manual_seed(701)
def rand(*shape,dtype=torch.float32):
    return torch.randn(shape,device='cuda',dtype=dtype)
h,t,m,k=12,72,16,48
for p in (216,432):
    b=h*t
    u=rand(b,m,8)
    co=rand(h,p,85,dtype=torch.bfloat16)
    a=rand(h,32,p,dtype=torch.bfloat16)
    z=rand(b,m,p,4)
    lm=rand(b,m,p,dtype=torch.float64)
    mean=rand(h,p,16,dtype=torch.float64)
    detail=rand(h,p,48,dtype=torch.float64)
    mask=rand(b,p)>0
    slots=rand(b,p).argsort(-1)[:,:k]
    mapping=torch.full((b,p),-1,device='cuda',dtype=torch.int32)
    mapping.scatter_(-1,slots,torch.arange(k,device='cuda',dtype=torch.int32).expand_as(slots))
    for size in (128,64,32):
        ns=triton.cdiv(p,size)
        def empty(*shape):return torch.empty(shape,device='cuda')
        state=empty(b,m,ns,3,16);logs=empty(b,m,ns,4);cand=empty(b,m,k,34);work=empty(b*m,ns,10,size)
        def split():
            scalar_kernel[(b*m,ns)](u,co,a,z,lm,mask,mapping,work,logs,cand,t,m,p,k,size,ns,num_warps=4,enable_fp_fusion=False)
            vector_kernel[(b*m,ns,1)](co,mean,detail,mask,mapping,work,state,cand,t,m,p,k,size,ns,num_warps=4,enable_fp_fusion=False)
        split()
        refs=[x.clone() for x in (state,logs,cand)]
        print(p,size,'split',triton.testing.do_bench_cudagraph(split,rep=50),flush=True)
        for warps in (2,4,8):
            def fused():
                return reader_kernel[(b*m,ns)](u,co,a,mean,detail,z,lm,mask,mapping,state,logs,cand,t,m,p,k,size,ns,True,False,True,num_warps=warps,enable_fp_fusion=False)
            kernel=fused()
            print(p,size,warps,kernel.n_regs,kernel.n_spills,*[(x-y).abs().nan_to_num().max().item() for x,y in zip(refs,(state,logs,cand))],triton.testing.do_bench_cudagraph(fused,rep=50),flush=True)
