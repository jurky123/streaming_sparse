"""PyTorch 2.6 FlexAttention with native per-query page masks.

Public implementation: https://pytorch.org/blog/flexattention/
The mask callback maps unpadded KV tokens back to per-chunk 64-token pages.
Forward-only block metadata is constructed directly (no dense token mask).
"""
import torch
import triton as tr
import triton.language as tl
from torch.nn.attention.flex_attention import flex_attention as flex,BlockMask


@tr.jit
def _physical_union(M,IDS,COUNTS,Q:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
                    G:tl.constexpr,T:tl.constexpr,P:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1)
    qi=g*G+tl.arange(0,G);pb=tl.arange(0,P);kb=tr.cdiv(K,T)
    first=pb*T;last=tl.minimum((pb+1)*T,K)-1
    start=first//4680*74+(first%4680)//64
    end=last//4680*74+(last%4680)//64
    use=tl.full((G,P),False,tl.int1)
    for j in tl.static_range(4):
        page=start+j
        v=tl.load(M+(h*Q+qi[:,None])*B+page[None,:],
            (qi[:,None]<Q)&(pb[None,:]<kb)&(page[None,:]<=end[None,:])&(page[None,:]<B),False)
        use=use|v
    used=tl.sum(use.to(tl.int32),0)>0
    rank=tl.cumsum(used.to(tl.int32),0)-1
    tl.store(IDS+(h*tr.cdiv(Q,G)+g)*kb+rank,pb,used)
    tl.store(COUNTS+h*tr.cdiv(Q,G)+g,tl.sum(used.to(tl.int32),0))


_compiled=torch.compile(flex,dynamic=False,fullgraph=True)


def flex_attention(q,k,v,mask,group=64,block=64):
    h,n,_=q.shape;nk=k.shape[1];kb=tr.cdiv(nk,block);qb=tr.cdiv(n,group)
    ids=torch.empty((1,h,qb,kb),device=q.device,dtype=torch.int32)
    counts=torch.empty((1,h,qb),device=q.device,dtype=torch.int32)
    _physical_union[(qb,h)](mask,ids,counts,n,nk,mask.shape[-1],group,block,tr.next_power_of_2(kb),num_warps=8)
    def mask_mod(batch,head,query,key):
        page=(key//4680)*74+(key%4680)//64
        return mask[head,query,page]
    # q metadata is only needed by backward; this adapter is inference-only.
    bm=BlockMask(seq_lengths=(n,nk),kv_num_blocks=counts,kv_indices=ids,
        full_kv_num_blocks=None,full_kv_indices=None,q_num_blocks=counts,q_indices=ids,
        full_q_num_blocks=None,full_q_indices=None,BLOCK_SIZE=(group,block),mask_mod=mask_mod)
    return _compiled(q[None],k[None],v[None],block_mask=bm,
        kernel_options={'BLOCK_M':group,'BLOCK_N':block})[0]
