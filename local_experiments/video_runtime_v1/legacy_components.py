"""Adapters for the previously optimized implementations; Full_Pset is unchanged."""
import torch
import triton as tr
from ..review_v1.fused_route import _normalizer,_risk
from ..review_v1.tuned_attention import _tail,_merge
from ..gap_exact_round_kernel import _attn_fwd
from ..high_sparse_runtime import _budget_mask


def risk(probes,z,summary,theta):
    # Reuse the current pipeline's QK logits instead of recomputing them.
    h,t,m,d=probes.shape;p=summary.mean.shape[1];n=t*m
    q=probes.bfloat16().contiguous();z=z.contiguous();st=summary.stats.contiguous();theta=theta.contiguous()
    lz=torch.empty((h,n),device=q.device);mom=torch.empty((h,n,9),device=q.device)
    bmax=torch.empty((h,),device=q.device);out=torch.empty((h,t,p),device=q.device)
    _normalizer[(n,h)](q,z,st,lz,mom,bmax,n,p,d,tr.next_power_of_2(p*4),tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False)
    _risk[(t,tr.cdiv(p,16),h)](z,lz,mom,st,theta,bmax,out,n,p,m,16,num_warps=4,enable_fp_fusion=False)
    return out

@torch.no_grad()
def attention(q,k,v,s,basis,ids,workspace,*,bf16_av=True):
    """Original page-LUT exact and mean/detail tail, directly on resident KV."""
    _,n,h,d=q.shape;p=k.shape[1]//64;count=ids.shape[-1]
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    def buf(name,shape,dtype=torch.float32):return workspace.tensor('legacy_'+name,shape,q.device,dtype)
    lut=ids.contiguous()[None];exact=buf('exact',q.shape);le=buf('le',(h,n))
    _attn_fwd[(n//64,h)](q,k,v,d**-.5,count,lut,le,exact,h,n,p*64,n//64,d,64,64,HIGH_PRECISION_AV=True,num_warps=4,num_stages=2)
    bits=buf('mask',(h,n//64,16),torch.uint32)
    _budget_mask[(n//64,h)](lut,bits,n//64,count,tr.next_power_of_2(count),num_warps=4)
    mu=buf('mu',(h,n,d));code=buf('code',(h,n,8));lt=buf('lt',(h,n));out=buf('out',q.shape,q.dtype)
    _tail[(n//64,h)](q,s.keys.contiguous(),s.mean.contiguous(),s.detail.contiguous(),bits,mu,code,lt,p,n,h,d,BQ=64,BR=64,BF16_AV=bf16_av,num_warps=4,num_stages=2,enable_fp_fusion=False)
    _merge[(n//16,h)](exact,le,mu,code,lt,basis.bfloat16().contiguous(),out,n,h,d,num_warps=4,enable_fp_fusion=False)
    return out
