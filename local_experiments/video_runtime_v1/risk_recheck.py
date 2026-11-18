"""Recompute full-precision page scores for query tiles with close top-k scores."""
import torch
import triton
import triton.language as tl
from .risk_online import _risk_tile
from .fixed_page_select import _select, _select_row


@triton.jit
def _rank(S,OWN,RANK,P:tl.constexpr,B:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0)
    i=tl.arange(0,B)
    x=tl.load(S+row*P+i,i<P,-float('inf'))
    own=tl.load(OWN+row)
    # Risk is nonnegative. Sort the unsigned encodings, including the forced own page.
    bits=tl.where(i==own,0x7f800000,tl.where(i<P,x.to(tl.uint32,bitcast=True),0))
    ranked=tl.sort(bits.to(tl.uint32),descending=True).to(tl.float32,bitcast=True)
    tl.store(RANK+row*(K+1)+i,ranked,i<=K)


@triton.jit
def _near(RANK,FLAG,K:tl.constexpr,B:tl.constexpr,GAP:tl.constexpr):
    row=tl.program_id(0)*16+tl.arange(0,16)
    i=tl.arange(0,B)
    ptr=RANK+row[:,None]*(K+1)+i[None,:]
    ranked=tl.load(ptr,i[None,:]<K,0)
    nxt=tl.load(ptr+1,i[None,:]<K,0)
    close=(i[None,:]>=1)&(i[None,:]<K)&((ranked-nxt)<=GAP*ranked)
    tl.store(FLAG+row,tl.max(close.to(tl.int32),1))


@triton.jit
def _compact(FLAG,ROWS,COUNT,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    flag=tl.load(FLAG+i,i<N,0)
    pos=tl.cumsum(flag,0)-1
    tl.store(ROWS+pos,i,flag!=0)
    tl.store(COUNT,tl.sum(flag,0))


@triton.jit
def _qk(Q,K,Z,ROWS,COUNT,T:tl.constexpr,P:tl.constexpr,WORK:tl.constexpr):
    block=tl.program_id(1)
    m=tl.arange(0,16)
    n=block*64+tl.arange(0,64)
    dk=tl.arange(0,32)
    for slot in range(tl.program_id(0),tl.load(COUNT),WORK):
        b=tl.load(ROWS+slot)
        h=b//T
        acc=tl.full((16,64),0,tl.float32)
        for j in range(4):
            d=j*32+dk
            q=tl.load(Q+(b*16+m[:,None])*128+d[None,:])
            k=tl.load(K+(h*P*4+n[None,:])*128+d[:,None],n[None,:]<P*4,0)
            acc=tl.dot(q,k,acc)
        tl.store(Z+(b*16+m[:,None])*P*4+n[None,:],acc*(128**-.5),n[None,:]<P*4)


@triton.jit
def _norm(Z,NORM,ROWS,COUNT,P:tl.constexpr,B:tl.constexpr,WORK:tl.constexpr):
    i=tl.arange(0,B)
    m=tl.program_id(1)
    for slot in range(tl.program_id(0),tl.load(COUNT),WORK):
        row=tl.load(ROWS+slot)*16+m
        z=tl.load(Z+row*P*4+i,i<P*4,-float('inf'))
        mx=tl.max(z,0)
        den=tl.sum(tl.exp(z-mx),0)
        tl.store(NORM+row*2,mx)
        tl.store(NORM+row*2+1,den)


@triton.jit
def _recheck(U,QN,Z,NORM,ST,TH,C,OUT,ROWS,COUNT,T:tl.constexpr,P:tl.constexpr,WORK:tl.constexpr):
    for slot in range(tl.program_id(0),tl.load(COUNT),WORK):
        b=tl.load(ROWS+slot)
        _risk_tile(b,U,QN,Z,NORM,ST,TH,C,OUT,T,16,P,16,False,True,True,False,True,0)


@triton.jit
def _reselect(S,OWN,IDS,MASK,ROWS,COUNT,P:tl.constexpr,K:tl.constexpr,B:tl.constexpr,WORK:tl.constexpr):
    for slot in range(tl.program_id(0),tl.load(COUNT),WORK):
        row=tl.load(ROWS+slot)
        _select_row(row,S,OWN,IDS,MASK,P,K,B)


def repair(q,k,u,qn,st,theta,c,own,out,select_pages=False,cuda_flags=False,radix_select=False):
    h,t,p=out.shape
    n=h*t
    flags=torch.empty(n,device=q.device,dtype=torch.int32)
    rows=torch.empty_like(flags)
    count=flags.new_empty(())
    logits=out.new_empty((n,16,4*p))
    norms=out.new_empty((n,16,2))
    top=p//10
    if not radix_select:
        ranked=out.new_empty((n,top+1))
    own=own.contiguous()
    if select_pages:
        ids=torch.empty((h,t,top),device=q.device,dtype=torch.int32)
        mask=torch.empty_like(out,dtype=torch.bool)
        if radix_select:
            from .risk_close import select_close
            select_close(out,own,ids,mask,rows,count)
        else:
            _select[(n,)](out,own,ids,mask,p,top,triton.next_power_of_2(p),ranked,num_warps=4)
    elif not cuda_flags:
        _rank[(n,)](out,own,ranked,p,triton.next_power_of_2(p),top,num_warps=4)
    if cuda_flags:
        from .risk_close import mark
        mark(out,own,rows,count)
    elif not radix_select:
        _near[(n//16,)](ranked,flags,top,triton.next_power_of_2(top),4.e-5,num_warps=4)
        _compact[(1,)](flags,rows,count,n,triton.next_power_of_2(n),num_warps=4)
    work=32
    _qk[(work,triton.cdiv(p*4,64))](q,k,logits,rows,count,t,p,work,num_warps=4)
    _norm[(work,16)](logits,norms,rows,count,p,triton.next_power_of_2(4*p),work,
                      num_warps=8 if p<=256 else 16,enable_fp_fusion=False)
    _recheck[(work,triton.cdiv(p,16))](u,qn,logits,norms,st,theta,c,out,rows,count,t,p,work,
                                       num_warps=8,enable_fp_fusion=False)
    if select_pages:
        _reselect[(work,)](out,own,ids,mask,rows,count,p,top,triton.next_power_of_2(p),work,num_warps=4)
        return ids,mask
    return count
