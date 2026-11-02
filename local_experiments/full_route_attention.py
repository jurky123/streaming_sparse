"""Native sparse+comp forward; FP32 memory-efficient SDPA recomputation backward."""
import math
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from .high_sparse_runtime import compensate

def native(q,k,v,pk,code,basis,center,lut,kernel,gamma):
    _,n,h,d=q.shape;out=torch.empty_like(q);lse=torch.empty(1,h,n,device=q.device)
    kernel[(n//64,h)](q,k,v,d**-.5,lut.shape[-1],lut,lse,out,h,n,k.shape[1],n//64,d,64,64,num_warps=4,num_stages=3)
    compensate(q,pk,code,lut,out,lse,basis,center,gamma)
    return out

def differentiable(q,k,v,pk,code,basis,center,lut):
    # Each tile/head has a compact sequence: selected real tokens, then omitted
    # K representatives with log(16) mass. No soft cross-page assignments.
    _,n,h,d=q.shape;t=n//64;p=pk.shape[1];keep=lut.shape[-1]
    ids=lut[0].long();heads=torch.arange(h,device=q.device)[:,None,None]
    token=(ids[...,None]*64+torch.arange(64,device=q.device)).flatten(-2)
    kk=k[0].transpose(0,1).float();vv=v[0].transpose(0,1).float()
    real_k=kk[heads,token];real_v=vv[heads,token]
    omitted=torch.ones(h,t,p,device=q.device,dtype=torch.bool).scatter_(-1,ids,False)
    pages=omitted.int().topk(p-keep,-1).indices.sort(-1).values
    rep_k=pk.float()[heads,pages].flatten(2,3)
    decoded=center.float()[:,None]+code.float()@basis.float().transpose(1,2)
    rep_v=decoded[heads,pages][:,:,:,None,:].expand(-1,-1,-1,4,-1).flatten(2,3)
    K=torch.cat((real_k,rep_k),2).reshape(h*t,-1,d);V=torch.cat((real_v,rep_v),2).reshape_as(K)
    Q=q[0].transpose(0,1).float().reshape(h*t,64,d)
    length=K.shape[1];padding=(-length)%8
    if padding:K=torch.nn.functional.pad(K,(0,0,0,padding));V=torch.nn.functional.pad(V,(0,0,0,padding))
    bias=torch.zeros(length+padding,device=q.device);bias[keep*64:length]=math.log(16)
    if padding:bias[length:]=-float('inf')
    bias=bias[None,None,None].expand(h*t,1,64,-1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out=torch.nn.functional.scaled_dot_product_attention(Q[:,None],K[:,None],V[:,None],attn_mask=bias,scale=d**-.5)
    return out[:,0].reshape(h,n,d).transpose(0,1)[None]

class NativeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,pk,code,basis,center,lut,kernel,gamma):
        ctx.save_for_backward(q,k,v,pk,code,basis,center,lut)
        return native(q,k,v,pk,code,basis,center,lut,kernel,gamma)
    @staticmethod
    def backward(ctx,dy):
        q,k,v,pk,code,basis,center,lut=ctx.saved_tensors;groups=[[],[],[],[],[]]
        # Bound recomputation working set; do not materialize H*T*Nhistory gradients.
        for start in range(0,q.shape[2],3):
            end=min(start+3,q.shape[2])
            with torch.enable_grad():
                xs=[q[:,:,start:end].detach().requires_grad_(True),k[:,:,start:end].detach().requires_grad_(True),v[:,:,start:end].detach().requires_grad_(True),pk[start:end].detach().requires_grad_(True),code[start:end].detach().requires_grad_(True)]
                out=differentiable(*xs,basis[start:end],center[start:end],lut[:,start:end])
                grads=torch.autograd.grad(out,xs,dy[:,:,start:end].float(),create_graph=False)
            for outlist,g in zip(groups,grads):outlist.append(g)
        grads=[torch.cat(g,2 if i<3 else 0) for i,g in enumerate(groups)]
        return *grads,None,None,None,None,None
