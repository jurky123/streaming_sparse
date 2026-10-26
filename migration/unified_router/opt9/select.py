"""Fuse threshold, deterministic candidate distances and hashes in registers."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _pick(rank, allowed, p, P:tl.constexpr, MODE:tl.constexpr):
    if MODE == "argmin":
        index=tl.argmin(tl.where(allowed,rank,float('inf')),0)
        valid=tl.sum(allowed.to(tl.int32),0)>0
        return tl.where(valid,index,2147483647)
    elif MODE == "exact32":
        bits=rank.to(tl.uint32,bitcast=True)
        best=tl.min(tl.where(allowed,bits,0x7fffffff),0)
        return tl.min(tl.where(allowed & (bits==best),p,2147483647),0)
    elif MODE == "packed64":
        key=(rank.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)|p.to(tl.uint64)
        best=tl.min(tl.where(allowed,key,0xffffffffffffffff),0)
        return tl.where(best!=0xffffffffffffffff,(best & 0xffffffff).to(tl.int32),2147483647)
    else:
        key=(rank.to(tl.uint32,bitcast=True) & (0xffffffff ^ (P-1)))|p.to(tl.uint32)
        best=tl.min(tl.where(allowed,key,0x7fffffff),0)
        return tl.where(best!=0x7fffffff,(best & (P-1)).to(tl.int32),2147483647)

@tr.jit
def _packed_select(X, S, O, REM, ADD, N: tl.constexpr, B: tl.constexpr, P: tl.constexpr, BETA:tl.constexpr, MODE:tl.constexpr):
    r = tl.program_id(0)
    p = tl.arange(0, P)
    x=tl.load(X+r*B+p,p<B,0.)
    mean=tl.sum(x,0)/B
    std=tl.sqrt(tl.sum(tl.where(p<B,(x-mean)*(x-mean),0.),0)/B)
    selected=(p<B)&((x>mean+BETA*std)|(std<=1.e-6))
    if tl.sum(selected.to(tl.int32),0)==0:
        maximum=tl.max(tl.where(p<B,x,float('-inf')),0)
        best=tl.min(tl.where((p<B)&(x==maximum),p,2147483647),0)
        selected=p==best
    tl.store(S+r*B+p,selected,p<B)
    origin=tl.load(O+p,p<B,-1)
    distance=tl.abs(x-mean-BETA*std)
    q=r%N;h=r//N
    tile_n=tl.minimum(128,N-q//128*128)
    canonical=h*tile_n+q%128
    raw=p.to(tl.uint64)*1103515245+canonical.to(tl.uint64)*12345+16
    folded=(raw & 2147483647)+(raw >> 31)
    hashes=tl.where(folded>=2147483647,folded-2147483647,folded).to(tl.float32)
    for side in tl.static_range(2):
        remain = (p < B) & (selected if side == 0 else ~selected)
        result = tl.full((8,), -1, tl.int32)
        cols = tl.arange(0, 8)
        for c in tl.static_range(8):
            allowed = remain
            if c < 3:
                allowed = allowed & (origin == 0)
            elif c < 6:
                allowed = allowed & (origin == 1)
            rank = distance if c < 6 else hashes
            idx = _pick(rank, allowed, p, P, MODE)
            valid = idx < 2147483647
            result = tl.where(cols == c, tl.where(valid, idx, -1), result)
            remain = remain & ~(valid & (p == idx))
        if tl.min(result, 0) < 0:
            for c in tl.static_range(8):
                missing = tl.sum(tl.where(cols == c, result, 0), 0) < 0
                idx = _pick(distance, remain, p, P, MODE)
                valid = missing & (idx < 2147483647)
                result = tl.where((cols == c) & valid, idx, result)
                remain = remain & ~(valid & (p == idx))
        if side == 0:
            tl.store(REM + r * 8 + cols, result)
        else:
            tl.store(ADD + r * 8 + cols, result)

def fused_select(score,origins,beta,mode="packed64"):
    h,n,b=score.shape
    mask=torch.empty_like(score,dtype=torch.bool)
    rem=torch.empty((h*n,8),device=score.device,dtype=torch.int64);add=torch.empty_like(rem)
    _packed_select[(h*n,)](score,mask,origins,rem,add,n,b,tr.next_power_of_2(b),beta,mode,num_warps=4,enable_fp_fusion=False)
    return mask,rem,add

