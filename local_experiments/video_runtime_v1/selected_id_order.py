"""Reconstruct CUDA topk's pre-sort order within the retained page IDs."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _prepare(RISK, IDS, VALUES, RAW, P:tl.constexpr,K:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    i=tl.arange(0,B)
    ids=tl.load(IDS+row*S+i,i<K,0)
    val=tl.load(RISK+row*P+ids)
    val=tl.where(i==0,float('inf'),val)
    cutoff=tl.min(tl.where(i<K,val,float('inf')),0)
    # CUDA gatherTopK emits strictly greater values in page order, then ties at the cutoff.
    order=tl.sort(tl.where(i<K,ids+tl.where(val==cutoff,P,0),2*P))
    raw=tl.minimum(order%P,P-1)
    v=tl.load(RISK+row*P+raw)
    own=tl.load(IDS+row*S)
    tl.store(VALUES+row*K+i,tl.where(raw==own,float('inf'),v),i<K)
    tl.store(RAW+row*K+i,raw,i<K)

@torch.library.custom_op('stream_sparse::compact_selected_order',mutates_args=())
def compact_order(risk:torch.Tensor,ids:torch.Tensor,count:int)->torch.Tensor:
    h,t,p=risk.shape
    val=risk.new_empty((h,t,count))
    raw=ids.new_empty((h,t,count))
    _prepare[(h*t,)](risk,ids,val,raw,p,count,ids.stride(1),tr.next_power_of_2(count))
    # The native small-sort tie order also controls the subsequent AV accumulation.
    return raw.gather(-1,val.sort(descending=True).indices)

@compact_order.register_fake
def _(risk,ids,count):
    return ids.new_empty((*ids.shape[:-1],count))

@tr.jit
def _tie_order(RISK,IDS,OUT,P:tl.constexpr,K:tl.constexpr,S:tl.constexpr,B:tl.constexpr,LOG:tl.constexpr):
    row=tl.program_id(0)
    i=tl.arange(0,B)
    ids=tl.load(IDS+row*S+i,i<K,0)
    own=tl.load(IDS+row*S)
    val=tl.where(ids==own,float('inf'),tl.load(RISK+row*P+ids))
    next_id=tl.load(IDS+row*S+i+1,i+1<K,0)
    nxt=tl.load(RISK+row*P+next_id)
    if tl.sum(((i<K-1)&(val==nxt)).to(tl.int32),0)>0:
        if K>32:
            # Native warp merge sort preserves input page order for equal scores.
            u=val.to(tl.uint32,bitcast=True)
            signed=tl.where((u>>31)!=0,~u,u^0x80000000)
            key=(signed.to(tl.uint64)<<32)|(P-ids).to(tl.uint64)
            sorted_keys=tl.sort(tl.where(i<K,key,0),descending=True)
            ids=P-(sorted_keys&0xffffffff).to(tl.int32)
        else:
            cutoff=tl.min(tl.where(i<K,val,float('inf')),0)
            raw=tl.sort(tl.where(i<K,ids+tl.where(val==cutoff,P,0),2*P))
            ids=tl.where(i<K,raw%P,-1)
            val=tl.where(ids==own,float('inf'),tl.load(RISK+row*P+ids,ids>=0,0))
            for level in tl.static_range(1,LOG+1):
                for step in tl.static_range(level):
                    stride=1<<(level-1-step)
                    ov=tl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",constraints="=f,f,r",args=[val,stride],dtype=tl.float32,is_pure=True,pack=1)
                    oi=tl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",constraints="=r,r,r",args=[ids,stride],dtype=tl.int32,is_pure=True,pack=1)
                    left=(i&stride)==0
                    before=tl.where(left,((val>ov)&(ids>=0))|(oi<0),((ov>val)&(oi>=0))|(ids<0))
                    swap=before==((i&(1<<level))!=0)
                    ids=tl.where(swap,oi,ids)
                    val=tl.where(swap,ov,val)
    tl.store(OUT+row*K+i,ids,i<K)

@torch.library.custom_op('stream_sparse::selected_tie_order',mutates_args=())
def tie_order(risk:torch.Tensor,ids:torch.Tensor,count:int)->torch.Tensor:
    out=ids.new_empty((*ids.shape[:-1],count))
    block=tr.next_power_of_2(count)
    _tie_order[(risk.shape[0]*risk.shape[1],)](risk,ids,out,risk.shape[-1],count,ids.stride(1),block,block.bit_length()-1,num_warps=1)
    return out

@tie_order.register_fake
def _(risk,ids,count):
    return ids.new_empty((*ids.shape[:-1],count))
