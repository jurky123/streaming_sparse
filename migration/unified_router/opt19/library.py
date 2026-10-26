"""Library attention on explicit full tiles. Planning is shape-only and reusable.

FlashInfer pages of length64 exclude each chunk's final8-token page in routing.
This approximation is recorded and tested, never treated as identical masks.
"""
import sys
from pathlib import Path
import torch
import triton as tr
import triton.language as tl
from ..opt14.flex_probe_gpu_meta import _pack,_transpose

@tr.jit
def _indices(IDS,OUT,T:tl.constexpr,K:tl.constexpr,B:tl.constexpr,ST:tl.constexpr,RATIO:tl.constexpr):
    z=tl.program_id(0)*256+tl.arange(0,256);h=tl.program_id(1)
    source=(z//K)//RATIO*K+z%K
    x=tl.load(IDS+h*ST*K+source,z<T*K,0)
    tl.store(OUT+h*T*K+z,x+h*B,z<T*K)

@tr.jit
def _query(Q,O,N:tl.constexpr,NN:tl.constexpr):
    z=tl.program_id(0)*256+tl.arange(0,256);h=tl.program_id(1)
    x=tl.load(Q+h*N*128+z,z<N*128,0)
    tl.store(O+h*NN*128+z,x,z<NN*128)

@tr.jit
def _used(IDS,U,T:tl.constexpr,K:tl.constexpr,B:tl.constexpr,KK:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1);j=tl.arange(0,KK)
    p=tl.load(IDS+(h*T+t)*K+j,j<K,0)
    tl.store(U+(h*T+t)*B+p,True,j<K)

def padded_kv(kv):
    k,v=kv.k,kv.v;h,n,d=k.shape;b=kv.state['key'].shape[1];nh=kv.history_tokens
    ko=torch.empty((h,b*64,128),device=k.device,dtype=k.dtype);vo=torch.empty_like(ko)
    kh,kc=k[:,:nh],k[:,nh:];vh,vc=v[:,:nh],v[:,nh:]
    _pack[(b,h)](kh,kc,vh,vc,ko,vo,b,nh,kh.stride(0),kh.stride(1),kc.stride(0),kc.stride(1),vh.stride(0),vh.stride(1),vc.stride(0),vc.stride(1),num_warps=4)
    return ko,vo


class FlashInfer:
    def __init__(self,h,n,b,group,keep,execution_group=None):
        deps=Path('streaming_sparse/v10/experiments/opt19_tile_kernels_20261025/deps').resolve()
        if str(deps) not in sys.path:sys.path.insert(0,str(deps))
        import flashinfer
        self.version=flashinfer.__version__;self.h=h;self.n=n;self.b=b
        self.group=group if execution_group is None else execution_group
        assert group%self.group==0
        self.ratio=group//self.group
        self.t=tr.cdiv(n,self.group);self.keep=keep;batch=h*self.t
        self.ids=torch.empty(batch*keep,device='cuda',dtype=torch.int32)
        qo=torch.arange(batch+1,device='cuda',dtype=torch.int32)*self.group
        ki=torch.arange(batch+1,device='cuda',dtype=torch.int32)*keep
        last=torch.full((batch,),64,device='cuda',dtype=torch.int32)
        self.ids.zero_()
        # Sequence lengths are FIXED here. The library's dynamic graph planner
        # assumes arbitrary future lengths and selects oversized query tiles.
        # Plan once with exact lengths; the resulting run remains graph-capturable.
        self.wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8),kv_layout='NHD',
            use_cuda_graph=False,backend='fa2')
        self.wrapper.plan(qo,ki,self.ids,last,1,1,128,64,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16)
        assert self.wrapper._paged_kv_indices_buf.data_ptr()==self.ids.data_ptr()

    def run(self,q,kv,ids):
        k,v=padded_kv(kv)
        return self.prepared(q,k,v,ids)

    def prepared(self,q,k,v,ids):
        h,n,d=q.shape;nn=self.t*self.group
        qp=torch.empty((h,nn,128),device=q.device,dtype=q.dtype)
        _query[(tr.cdiv(nn*128,256),h)](q,qp,n,nn,num_warps=4)
        _indices[(tr.cdiv(self.t*self.keep,256),h)](ids,self.ids,self.t,self.keep,self.b,ids.shape[1],self.ratio,num_warps=4)
        out=self.wrapper.run(qp.reshape(h*nn,1,128),(k.reshape(h*self.b,64,1,128),v.reshape(h*self.b,64,1,128)))
        return out.reshape(h,nn,128)[:,:n]


class Flex:
    def __init__(self,group):
        from torch.nn.attention.flex_attention import flex_attention
        import importlib
        impl=importlib.import_module('torch._inductor.kernel.flex_attention')
        impl._get_default_config_fwd=lambda query:(group,64,4,3)
        self.compiled=torch.compile(flex_attention,dynamic=False);self.group=group

    def run(self,q,kv,ids):
        k,v=padded_kv(kv)
        return self.prepared(q,k,v,ids)

    def prepared(self,q,k,v,ids):
        from torch.nn.attention.flex_attention import BlockMask
        h,n,d=q.shape;b=k.shape[1]//64;g=self.group;t=ids.shape[1];s=ids.shape[2]
        used=torch.zeros((h,t,b),device=q.device,dtype=torch.bool)
        _used[(t,h)](ids,used,t,s,b,tr.next_power_of_2(s),num_warps=4)
        qids=torch.empty((h,b,t),device=q.device,dtype=torch.int32)
        qcounts=torch.empty((h,b),device=q.device,dtype=torch.int32)
        _transpose[(b,h)](used,qids,qcounts,t,b,tr.next_power_of_2(t),num_warps=4)
        counts=torch.full((1,h,t),s,device=q.device,dtype=torch.int32)
        zc=torch.zeros_like(counts);zq=torch.zeros_like(qcounts[None])
        def visible(batch,head,qi,ki):return qi>=0
        bm=BlockMask((n,b*64),zc,ids[None],counts,ids[None],zq,qids[None],qcounts[None],qids[None],(g,64),visible)
        return self.compiled(q[None],k[None],v[None],block_mask=bm,
                             kernel_options={'BLOCK_M':g,'BLOCK_N':64})[0]
