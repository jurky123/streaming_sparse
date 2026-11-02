"""Independent explicit softmax checks for all attention input gradient paths."""
import math,json
import torch
from .full_route_attention import NativeAttention,differentiable
from .cache_sparse_prototype import load_kernel
from .learned_cache.io import atomic

def reference(q,k,v,pk,code,basis,center,lut):
    _,n,h,d=q.shape;p=pk.shape[1];mask=torch.zeros(h,n//64,p,device=q.device,dtype=torch.bool).scatter_(-1,lut[0].long(),True).repeat_interleave(64,1)
    Q=q[0].transpose(0,1).float();K=k[0].transpose(0,1).float();V=v[0].transpose(0,1).float()
    a=(Q@K.transpose(1,2)/math.sqrt(d)).masked_fill(~mask.repeat_interleave(64,-1),-float('inf'))
    b=(Q@pk.float().flatten(1,2).transpose(1,2)/math.sqrt(d)+math.log(16)).masked_fill(mask.repeat_interleave(4,-1),-float('inf'))
    decoded=center[:,None]+code.float()@basis.float().transpose(1,2)
    values=torch.cat((V,decoded.repeat_interleave(4,1)),1)
    return (torch.cat((a,b),-1).softmax(-1)@values).transpose(0,1)[None]

def main():
    torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    rows=[];kernel=load_kernel()
    for p,keep,n in [(6,2,128),(9,3,64)]:
        h=3;d=128;device='cuda'
        values=[torch.randn(1,n,h,d,device=device,dtype=torch.bfloat16),torch.randn(1,p*64,h,d,device=device,dtype=torch.bfloat16),torch.randn(1,p*64,h,d,device=device,dtype=torch.bfloat16),torch.randn(h,p,4,d,device=device)*.2,torch.randn(h,p,8,device=device,dtype=torch.bfloat16)]
        xs=[x.requires_grad_() for x in values];basis=torch.linalg.qr(torch.randn(h,d,8,device=device))[0].to(torch.bfloat16).contiguous();center=torch.randn(h,d,device=device)*.1
        lut=torch.rand(h,n//64,p,device=device).topk(keep,-1).indices.sort(-1).values[None].int();dy=torch.randn_like(xs[0])
        y=NativeAttention.apply(*xs,basis,center,lut,kernel,torch.ones(h,device=device));gs=torch.autograd.grad(y,xs,dy)
        yr=reference(*xs,basis,center,lut);gr=torch.autograd.grad(yr,xs,dy.float())
        forward=float((y.float()-yr).square().sum()/yr.square().sum());assert forward<1e-4,forward
        errors=[]
        for g,r in zip(gs,gr):
            e=float((g.float()-r.float()).square().sum()/r.float().square().sum().clamp_min(1e-12));assert e<2e-5,e;assert torch.isfinite(g).all();errors.append(e)
        rows.append(dict(pages=p,keep=keep,queries=n,forward_nmse=forward,gradient_nmse=dict(zip(['Q','K','V','Kprototype','Vcode'],errors))))
    atomic('artifacts/route_graph_20261031/backward_checks.json',dict(passed=True,rows=rows));print(json.dumps(dict(passed=True,rows=rows)),flush=True)
if __name__=='__main__':main()
