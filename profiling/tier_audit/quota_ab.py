import torch
import triton
def old(x,k):
    order=x.argsort(dim=-1,descending=True,stable=True)
    return order.argsort(-1)<k
def new(x,k):
    order=x.argsort(dim=-1,descending=True,stable=True)
    return torch.empty_like(x,dtype=torch.bool).scatter_(-1,order,(torch.arange(x.shape[-1],device=x.device)<k).expand_as(x))
if __name__=='__main__':
    a,b=map(torch.compile,(old,new))
    for n in (72,120,192,264,336):
        x=torch.randint(0,32,(12,72,n),device='cuda').float()
        k=torch.randint(0,n+1,(12,1,1),device='cuda')
        assert torch.equal(a(x,k),b(x,k))
        print(n,*[triton.testing.do_bench_cudagraph(lambda:f(x,k),rep=30) for f in (a,b)],flush=True)
