"""Independent explicit-softmax audit of all nine conditional attention gradients."""
import json,math,torch
from .all_sparse_attention import Attention
from .cache_sparse_prototype import load_kernel
from .learned_cache.io import atomic

def reference(q,k,v,pk,base,detail,basis,center,scale,lut):
    _,n,h,d=q.shape;p=pk.shape[1]
    mask=torch.zeros(h,n//64,p,device=q.device,dtype=torch.bool).scatter_(-1,lut[0].long(),True).repeat_interleave(64,1)
    Q=q[0].transpose(0,1).float();K=k[0].transpose(0,1).float();V=v[0].transpose(0,1).float()
    a=(Q@K.transpose(1,2)/math.sqrt(d)).masked_fill(~mask.repeat_interleave(64,-1),-float('inf'))
    b=(Q@pk.float().flatten(1,2).transpose(1,2)/math.sqrt(d)+math.log(16)+scale[:,None,None]).masked_fill(mask.repeat_interleave(4,-1),-float('inf'))
    # Explicit four conditional values, independent of reconstruct().
    u,w,z=detail.float().unbind(2);c=base.float()
    codes=torch.stack((c+u+w+z,c+u-w-z,c-u+w-z,c-u-w+z),2)
    decoded=center[:,None,None]+codes@basis.float()[:,None].transpose(-1,-2)
    return (torch.cat((a,b),-1).softmax(-1)@torch.cat((V,decoded.flatten(1,2)),1)).transpose(0,1)[None]

def main():
    torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    rows=[];kernel=load_kernel()
    for p,keep,n in [(6,2,128),(9,3,64)]:
        h=3;d=128
        def rand(*s):return torch.randn(*s,device='cuda')
        xs=[rand(1,n,h,d).bfloat16(),rand(1,p*64,h,d).bfloat16(),rand(1,p*64,h,d).bfloat16(),rand(h,p,4,d)*.2,rand(h,p,8).bfloat16(),(rand(h,p,3,8)*.1).bfloat16(),torch.linalg.qr(rand(h,d,8))[0].bfloat16().contiguous(),rand(h,d)*.1,rand(h)*.1]
        xs=[x.requires_grad_() for x in xs];lut=rand(h,n//64,p).topk(keep,-1).indices.sort(-1).values[None].int();dy=torch.randn_like(xs[0])
        y=Attention.apply(*xs,lut,kernel);gs=torch.autograd.grad(y,xs,dy)
        yr=reference(*xs,lut);gr=torch.autograd.grad(yr,xs,dy.float())
        nmse=float((y.float()-yr).square().sum()/yr.square().sum());assert nmse<1e-4,nmse
        errors={}
        for name,g,r in zip(['q','k','v','pk','base','detail','basis','center','logscale'],gs,gr):
            e=float((g.float()-r.float()).square().sum()/r.float().square().sum().clamp_min(1e-12));assert torch.isfinite(g).all() and g.norm()>0;assert e<3e-5,(name,e);errors[name]=e
        rows.append(dict(pages=p,forward_nmse=nmse,gradient_nmse=errors))
    result=dict(passed=True,rows=rows);atomic('artifacts/sparse_expansion_20261031/all_sparse_backward_checks.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
