"""Native conditional-V forward and full conditional gradients, including sparse parameters."""
import math
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from .conditional_v_kernels import compensate,reconstruct

def native(q,k,v,pk,base,detail,basis,center,scale,lut,kernel):
    _,n,h,d=q.shape;out=torch.empty_like(q);lse=torch.empty(1,h,n,device=q.device)
    kernel[(n//64,h)](q,k,v,d**-.5,lut.shape[-1],lut,lse,out,h,n,k.shape[1],n//64,d,64,64,num_warps=4,num_stages=3)
    compensate(q,pk,base,detail,lut,lut,out,lse,basis,center,torch.ones(h,device=q.device),scale)
    return out

def differentiable(q,k,v,pk,base,detail,basis,center,scale,lut):
    _,n,h,d=q.shape;t=n//64;p=pk.shape[1];keep=lut.shape[-1];ids=lut[0].long();heads=torch.arange(h,device=q.device)[:,None,None]
    token=(ids[...,None]*64+torch.arange(64,device=q.device)).flatten(-2)
    kk=k[0].transpose(0,1).float();vv=v[0].transpose(0,1).float();real_k=kk[heads,token];real_v=vv[heads,token]
    omit=torch.ones(h,t,p,device=q.device,dtype=torch.bool).scatter_(-1,ids,False);pages=omit.int().topk(p-keep,-1).indices.sort(-1).values
    rep_k=pk.float()[heads,pages].flatten(2,3);codes=reconstruct(base,detail);decoded=center[:,None,None]+torch.einsum('hprk,hdk->hprd',codes,basis.float())
    rep_v=decoded[heads,pages].flatten(2,3)
    K=torch.cat((real_k,rep_k),2).reshape(h*t,-1,d);V=torch.cat((real_v,rep_v),2).reshape_as(K);Q=q[0].transpose(0,1).float().reshape(h*t,64,d)
    length=K.shape[1];pad=(-length)%8
    if pad:K=torch.nn.functional.pad(K,(0,0,0,pad));V=torch.nn.functional.pad(V,(0,0,0,pad))
    real_bias=torch.zeros(h,t,keep*64,device=q.device);rep_bias=(math.log(16)+scale[:,None,None]).expand(h,t,(p-keep)*4)
    bias=torch.cat((real_bias,rep_bias),-1)
    if pad:bias=torch.nn.functional.pad(bias,(0,pad),value=-float('inf'))
    bias=bias.reshape(h*t,1,1,-1).expand(-1,1,64,-1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):out=torch.nn.functional.scaled_dot_product_attention(Q[:,None],K[:,None],V[:,None],attn_mask=bias,scale=d**-.5)
    return out[:,0].reshape(h,n,d).transpose(0,1)[None]

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,pk,base,detail,basis,center,scale,lut,kernel):
        ctx.save_for_backward(q,k,v,pk,base,detail,basis,center,scale,lut)
        return native(q,k,v,pk,base,detail,basis,center,scale,lut,kernel)
    @staticmethod
    def backward(ctx,dy):
        *saved,lut=ctx.saved_tensors;groups=[[] for _ in saved]
        for start in range(0,saved[0].shape[2],3):
            end=min(start+3,saved[0].shape[2])
            with torch.enable_grad():
                xs=[(x[:,:,start:end] if i<3 else x[start:end]).detach().requires_grad_(True) for i,x in enumerate(saved)]
                out=differentiable(*xs,lut[:,start:end]);grads=torch.autograd.grad(out,xs,dy[:,:,start:end].float(),create_graph=False)
            for parts,g in zip(groups,grads):parts.append(g)
        return *[torch.cat(parts,2 if i<3 else 0) for i,parts in enumerate(groups)],None,None
