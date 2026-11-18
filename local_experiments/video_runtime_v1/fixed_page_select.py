"""Select the fixed 10% page set using the stable capacity-sort ordering."""
import torch
import triton
import triton.language as tl


@triton.jit
def _select_row(row,S,OWN,IDS,MASK,P:tl.constexpr,K:tl.constexpr,B:tl.constexpr,RANK=None):
    i=tl.arange(0,B)
    own=tl.load(OWN+row).to(tl.int32)
    value=tl.load(S+row*P+i,i<P,0)
    value=tl.where(i==own,float('inf'),value)
    # CUDA gatherTopK uses the radix representation, including the zero sign.
    bits=value.to(tl.uint32,bitcast=True)
    ordered=tl.where((bits>>31)!=0,~bits,bits^0x80000000)
    key=(ordered.to(tl.uint64)<<32)|(P-i).to(tl.uint32).to(tl.uint64)
    key=tl.where(i<P,key,0)
    sorted_keys=tl.sort(key,descending=True)
    cutoff=tl.sum(tl.where(i==K-1,sorted_keys,0),0)
    tl.store(MASK+row*P+i,key>=cutoff,i<P)
    tl.store(IDS+row*K+i,P-(sorted_keys&0xffffffff).to(tl.int32),i<K)

    if RANK is not None:
        high=(sorted_keys>>32).to(tl.uint32)
        bits=tl.where((high>>31)!=0,high^0x80000000,~high)
        tl.store(RANK+row*(K+1)+i,bits.to(tl.float32,bitcast=True),i<=K)


@triton.jit
def _select(S,OWN,IDS,MASK,P:tl.constexpr,K:tl.constexpr,B:tl.constexpr,RANK=None):
    _select_row(tl.program_id(0),S,OWN,IDS,MASK,P,K,B,RANK)


@torch.library.custom_op('stream_sparse::fixed_page_select',mutates_args=())
def select(scores:torch.Tensor,own:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    h,t,p=scores.shape
    k=p//10
    ids=torch.empty((h,t,k),device=scores.device,dtype=torch.int32)
    mask=torch.empty_like(scores,dtype=torch.bool)
    _select[(h*t,)](scores,own.contiguous(),ids,mask,p,k,triton.next_power_of_2(p),num_warps=4)
    return ids,mask


@select.register_fake
def _(scores,own):
    return (scores.new_empty((*scores.shape[:-1],scores.size(-1)//10),dtype=torch.int32),
            torch.empty_like(scores,dtype=torch.bool))
