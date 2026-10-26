"""Inference kernels. Original router/training files remain untouched.

Attention uses the online-softmax algorithm described by FlashAttention and
Triton's fused-attention tutorial, with per-query page masks and segment tails.
No third-party source is vendored. TF32 is never used by these kernels.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _candidates(X, S, O, DIST, HASH, REM, ADD, N:tl.constexpr, B:tl.constexpr, P:tl.constexpr):
    r=tl.program_id(0); p=tl.arange(0,P)
    selected=tl.load(S+r*B+p,p<B,False)
    origin=tl.load(O+p,p<B,-1)
    distance=tl.load(DIST+r*B+p,p<B,float('inf'))
    hashes=tl.load(HASH+r*B+p,p<B,float('inf'))
    for side in tl.static_range(2):
        remain=(p<B)&(selected if side==0 else ~selected)
        # Pick stable minima: equal ranks always prefer the lowest page index.
        result=tl.full((8,),-1,tl.int32); cols=tl.arange(0,8)
        for c in tl.static_range(8):
            allowed=remain
            if c<3: allowed=allowed&(origin==0)
            elif c<6: allowed=allowed&(origin==1)
            rank=distance if c<6 else hashes
            best=tl.min(tl.where(allowed,rank,float('inf')),0)
            idx=tl.min(tl.where(allowed&(rank==best),p,2147483647),0)
            valid=idx<2147483647
            result=tl.where(cols==c,tl.where(valid,idx,-1),result)
            remain=remain&~(valid&(p==idx))
        for c in tl.static_range(8):
            missing=tl.sum(tl.where(cols==c,result,0),0)<0
            best=tl.min(tl.where(remain,distance,float('inf')),0)
            idx=tl.min(tl.where(remain&(distance==best),p,2147483647),0)
            valid=missing&(idx<2147483647)
            result=tl.where((cols==c)&valid,idx,result)
            remain=remain&~(valid&(p==idx))
        if side==0: tl.store(REM+r*8+cols,result)
        else: tl.store(ADD+r*8+cols,result)


def candidate_sides(score,mask,origins,beta,canonical_tile=128,query_stride=1,query_count=None):
    h,n,b=score.shape
    std,mean=torch.std_mean(score,dim=-1,unbiased=False,keepdim=True)
    distance=(score-mean-beta*std).abs()
    # Preserve the original hash reset for every canonical 128-query tile,
    # including its last (72-query) tile. Larger launch batches are only scheduling.
    q=torch.arange(n,device=score.device)*query_stride
    query_count=n if query_count is None else query_count
    tile_n=torch.minimum(torch.full_like(q,canonical_tile), query_count-(q//canonical_tile)*canonical_tile)
    row=torch.arange(h,device=score.device)[:,None]*tile_n[None]+(q%canonical_tile)[None]
    hashes=((torch.arange(b,device=score.device)*1103515245+row[...,None]*12345+16)%2147483647).float()
    rem=torch.empty((h*n,8),device=score.device,dtype=torch.int64);add=torch.empty_like(rem)
    _candidates[(h*n,)](score,mask,origins,distance,hashes,rem,add,h*n,b,tr.next_power_of_2(b),num_warps=4)
    return rem,add


@tr.jit
def _union(M, IDS, COUNTS, Q:tl.constexpr,B:tl.constexpr,G:tl.constexpr,P:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1)
    qi=g*G+tl.arange(0,G);p=tl.arange(0,P)
    m=tl.load(M+(h*Q+qi[:,None])*B+p[None,:],(qi[:,None]<Q)&(p[None,:]<B),False)
    used=tl.sum(m.to(tl.int32),0)>0
    rank=tl.cumsum(used.to(tl.int32),0)-1
    tl.store(IDS+(h*tr.cdiv(Q,G)+g)*B+rank,p,used)
    tl.store(COUNTS+h*tr.cdiv(Q,G)+g,tl.sum(used.to(tl.int32),0))


@tr.jit
def _attention(QP,KP,VP,M,IDS,COUNTS,OUT,
               Q:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
               QH:tl.constexpr,QQ:tl.constexpr,KH:tl.constexpr,KK:tl.constexpr,
               VH:tl.constexpr,VK:tl.constexpr,G:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1)
    qi=g*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,64)
    q=tl.load(QP+h*QH+qi[:,None]*QQ+d[None,:],qi[:,None]<Q,0)
    maximum=tl.full((G,),float('-inf'),tl.float32)
    denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
    count=tl.load(COUNTS+h*tr.cdiv(Q,G)+g)
    for j in range(count):
        page=tl.load(IDS+(h*tr.cdiv(Q,G)+g)*B+j)
        # 4680-token segments have 73 full pages and one 8-token tail.
        segment=page//74;local=page%74;ki=segment*4680+local*64+t
        valid=(ki<K)&(local*64+t<4680)
        k=tl.load(KP+h*KH+ki[None,:]*KK+d[:,None],valid[None,:],0)
        logits=tl.dot(q,k).to(tl.float32)*0.12751743082459617 # log2(e)/sqrt(128)
        visible=tl.load(M+(h*Q+qi)*B+page,qi<Q,False)
        logits=tl.where(visible[:,None]&valid[None,:],logits,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(logits,1))
        safe=tl.where(newmax==float('-inf'),0.,newmax)
        alpha=tl.exp2(maximum-safe)
        prob=tl.exp2(logits-safe[:,None])
        v=tl.load(VP+h*VH+ki[:,None]*VK+d[None,:],valid[:,None],0)
        acc=acc*alpha[:,None]+tl.dot(prob.to(v.dtype),v)
        denom=denom*alpha+tl.sum(prob,1);maximum=newmax
    tl.store(OUT+(h*Q+qi[:,None])*128+d[None,:],acc/denom[:,None],qi[:,None]<Q)


def sparse_attention(q,k,v,mask,group=16,return_union=False):
    h,n,d=q.shape;b=mask.shape[-1]
    assert d==128 and q.dtype==k.dtype==v.dtype==torch.bfloat16
    ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32)
    counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
    _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    _attention[(tr.cdiv(n,group),h)](q,k,v,mask,ids,counts,out,n,k.shape[1],b,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),group,
        num_warps=4,num_stages=2)
    return (out,counts) if return_union else out
