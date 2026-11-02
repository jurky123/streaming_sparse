import torch,triton as tr,triton.language as tl
@tr.jit
def dot(F,W,O,N:tl.constexpr,ACC:tl.constexpr,FMA:tl.constexpr):
    x=tl.program_id(0)*256+tl.arange(0,256);i=tl.arange(0,ACC)
    sums=tl.full((256,ACC),0,tl.float32)
    for a in range(tr.cdiv(14,ACC)):
        j=a*ACC+i
        f=tl.load(F+x[:,None]*14+j[None,:],(x[:,None]<N)&(j[None,:]<14),0);w=tl.load(W+j,j<14,0)
        if FMA:sums=tl.fma(f,w[None,:],sums)
        else:sums=sums+f*w[None,:]
    out=tl.sum(sums,1);tl.store(O+x,out,x<N)
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    e={k:v.cuda() for k,v in torch.load('artifacts/route_kernel_20261031/debug/boundary.pt',weights_only=True).items()}
    f=e['sf'];w=e['w'];ref=torch.einsum('htpf,hf->htp',f,w);out=torch.empty_like(ref)
    for acc in [1,2,4,8,16]:
        for fma in [False,True]:
            dot[(tr.cdiv(out.numel(),256),)](f,w,out,out.numel(),acc,fma,enable_fp_fusion=False)
            print(acc,fma,int((out!=ref).sum()),float((out-ref).abs().max()),flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(2):torch.einsum('htpf,hf->htp',f,w)
        torch.cuda.synchronize()
    print([e.key for e in prof.key_averages() if 'gem' in e.key])
if __name__=='__main__':main()
