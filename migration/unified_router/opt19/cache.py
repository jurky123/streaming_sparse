"""Owned grouped KV workspace: update current chunk without recopying history."""
import torch
import triton as tr
import triton.language as tl
from ..opt16.history import HistoryBuilder
from ..opt16.grouping import order
from ..opt16.route import prepare
from ..opt16.pipeline import PackedKV
from .prototype import segment

@tr.jit
def _write(K,V,IX,KO,VO,N:tl.constexpr,NK:tl.constexpr,OFFSET:tl.constexpr,
           KH:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VT:tl.constexpr):
    h=tl.program_id(1);z=tl.program_id(0)*256+tl.arange(0,256)
    t=z//128;d=z%128;ix=tl.load(IX+h*N+t,t<N,0)
    k=tl.load(K+h*KH+ix*KT+d,t<N,0);v=tl.load(V+h*VH+ix*VT+d,t<N,0)
    dst=(h*NK+OFFSET+t)*128+d
    tl.store(KO+dst,k,t<N);tl.store(VO+dst,v,t<N)


class Cache:
    def __init__(self,pipeline,raw,precision='tf32x3'):
        self.pipeline=pipeline;self.precision=precision
        self.builder=HistoryBuilder(pipeline)
        if precision!='reference':
            self.builder.segment=lambda k,w,b,p:segment(k,w,b,p,half=precision=='fp16')
        self.history=self.builder.history(raw['kh'],raw['vh'])
        self.nh=raw['kh'].shape[1];self.n=raw['kc'].shape[1]
        h,_,d=raw['kh'].shape
        self.k=torch.empty((h,self.nh+self.n,d),device=raw['kh'].device,dtype=raw['kh'].dtype)
        self.v=torch.empty_like(self.k)
        self.k[:,:self.nh].copy_(self.history.k);self.v[:,:self.nh].copy_(self.history.v)

    def refresh_history(self,k,v):
        if k.shape[1]!=self.nh or k.shape!=v.shape:raise ValueError('Create a new cache when history shape changes')
        self.history=self.builder.history(k,v)
        self.k[:,:self.nh].copy_(self.history.k);self.v[:,:self.nh].copy_(self.history.v)

    def current(self,k,v):
        par=self.pipeline.parameters
        ix=order(k,self.pipeline.ktree)
        _write[(tr.cdiv(self.n*128,256),k.shape[0])](k,v,ix,self.k,self.v,self.n,self.nh+self.n,self.nh,
             k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=4)
        pp=self.builder.segment(self.k[:,self.nh:],par['current_weight'],par['current_bias'],par['key_basis'])
        state=prepare((par['key_basis'][...,:64],list(self.history.parts)+[pp]))
        return PackedKV(self.k,self.v,None,state,self.nh)
