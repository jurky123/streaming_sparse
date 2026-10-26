"""Fuse exact data preparation around unchanged Opt26 routing/attention."""
import torch,triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from ..opt16.grouping import order
from ..opt16.pipeline import PackedKV
from ..opt19.prototype import segment
from ..opt16.route import prepare
@tr.jit
def _q(Q,IX,BASIS,P,QI,QS,N:tl.constexpr,T:tl.constexpr,QH:tl.constexpr,QN:tl.constexpr,BH:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);r=g*32+tl.arange(0,32);d=tl.arange(0,128);c=tl.arange(0,64)
 ix=tl.load(IX+h*N+r,r<N,0);q=tl.load(Q+h*QH+ix[:,None]*QN+d[None,:],r[:,None]<N,0)
 basis=tl.load(BASIS+h*BH+d[:,None]*BK+c[None,:]*BD).to(tl.float16)
 pp=tl.dot(q.to(tl.float16),basis);tl.store(P+(h*N+r[:,None])*64+c[None,:],pp,r[:,None]<N)
 qf=q.to(tl.float32);scale=tl.maximum(tl.max(tl.max(tl.abs(qf),1),0)/127.,1e-10)
 tl.store(QI+(h*N+r[:,None])*128+d[None,:],nearbyint(qf/scale).to(tl.int8),r[:,None]<N);tl.store(QS+h*T+g,scale)

def query(q,ix,basis):
 h,n,d=q.shape;t=tr.cdiv(n,32);proj=torch.empty((h,n,64),device=q.device,dtype=torch.float16);qi=torch.empty((h,n,128),device=q.device,dtype=torch.int8);qs=torch.empty((h,t),device=q.device)
 _q[(t,h)](q,ix,basis,proj,qi,qs,n,t,q.stride(0),q.stride(1),basis.stride(0),basis.stride(1),basis.stride(2),num_warps=4,num_stages=1)
 return proj,qi,qs

@tr.jit
def _kv(K,V,IX,KO,VO,KI,KS,PK,PV,N:tl.constexpr,NH:tl.constexpr,NT:tl.constexpr,B:tl.constexpr,HP:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,VH:tl.constexpr,VN:tl.constexpr):
 page=tl.program_id(0);h=tl.program_id(1);r=page*64+tl.arange(0,64);d=tl.arange(0,128)
 ix=tl.load(IX+h*N+r,r<N,0);k=tl.load(K+h*KH+ix[:,None]*KN+d[None,:],r[:,None]<N,0);v=tl.load(V+h*VH+ix[:,None]*VN+d[None,:],r[:,None]<N,0)
 old=(h*NT+NH+r[:,None])*128+d[None,:];packed=((h*B+HP+page)*64+tl.arange(0,64)[:,None])*128+d[None,:]
 tl.store(KO+old,k,r[:,None]<N);tl.store(VO+old,v,r[:,None]<N)
 scale=tl.maximum(tl.max(tl.max(tl.abs(k.to(tl.float32)),1),0)/127.,1e-10);ki=nearbyint(k.to(tl.float32)/scale).to(tl.int8)
 tl.store(KI+old,ki,r[:,None]<N);tl.store(KS+h*B+HP+page,scale);tl.store(PK+packed,ki);tl.store(PV+packed,v.to(tl.float16))

def current(pipe,k,v):
 cache=pipe.cache;p=cache.par;ix=order(k,pipe.model.tree('k'));h,n,d=k.shape
 _kv[(tr.cdiv(n,64),h)](k,v,ix,cache.k,cache.v,pipe.quant.k,pipe.quant.scale,pipe.pk,pipe.pv,n,cache.nh,cache.nh+n,pipe.quant.b,pipe.quant.nhpages,k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=4,num_stages=1)
 pp=segment(cache.k[:,cache.nh:],p['current_weight'],p['current_bias'],p['key_basis'],half=True)
 return PackedKV(cache.k,cache.v,None,prepare((p['key_basis'][...,:64],list(cache.history.parts)+[pp])),cache.nh)
