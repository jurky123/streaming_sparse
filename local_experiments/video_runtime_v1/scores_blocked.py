"""Coalesce action loads across candidates; skip fallback in groups on the GPU."""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .scores_packed import packed_kernel

@tr.jit
def scalar_block(DEN,STATS,DOT,MAX,OUT,FLAGS,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr,BLOCK:tl.constexpr):
    b=tl.program_id(1);action=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ri=action//A;ai=action%A;m=tl.arange(0,M)
    ix=(b*M+m[:,None])*(R*A)+action[None,:];valid=action<R*A
    den=tl.load(DEN+ix,valid[None,:],0);maximum=tl.load(MAX+b*M+m)[:,None]
    ir=((b*M+m[:,None])*(R+A)+ri[None,:])*2;ia=((b*M+m[:,None])*(R+A)+R+ai[None,:])*2
    er=tl.load(STATS+ir,valid[None,:],0);nr=tl.load(STATS+ir+1,valid[None,:],0)
    ea=tl.load(STATS+ia,valid[None,:],0);na=tl.load(STATS+ia+1,valid[None,:],0)
    dot=tl.load(DOT+ix,valid[None,:],0);norm=nr+na-2*dot;linear=ea-er
    scale=lib.exp(maximum-den);first=2*scale*linear;second=scale*scale*norm;g=first-second
    bad=(maximum-den>80)|(tl.abs(g)>=float('inf'))|(g!=g)|(norm<0)
    bad=bad|((nr+na>0)&(tl.abs(norm)<1e-8*(nr+na)))
    bad=bad|((tl.abs(er)+tl.abs(ea)>0)&(tl.abs(linear)<1e-8*(tl.abs(er)+tl.abs(ea))))
    mag=tl.abs(first)+tl.abs(second);bad=bad|((mag>0)&(tl.abs(g)<1e-8*mag))
    total=tl.sum(g,0);absolute=tl.sum(tl.abs(g),0)
    fallback=(tl.sum(bad.to(tl.int32),0)>0)|((absolute>0)&(tl.abs(total)<1e-8*absolute))
    tl.store(OUT+b*R*A+action,total/M,valid);tl.store(FLAGS+b*R*A+action,fallback,valid)

@tr.jit
def fallback_block(DEN,STATE,CAND,LB,RX,RO,AX,AO,MAX,OUT,FLAGS,N:tl.constexpr,R:tl.constexpr,A:tl.constexpr,M:tl.constexpr,D:tl.constexpr,G:tl.constexpr,LAZY:tl.constexpr=False):
    base=tl.program_id(0)*G;i=tl.arange(0,G);flags=tl.load(FLAGS+base+i,base+i<N,0)
    if tl.sum(flags.to(tl.int32),0)>0:
        for j in range(G):
            action=base+j
            if tl.load(FLAGS+action,action<N,0):
                packed_kernel(DEN,STATE,CAND,LB,RX,RO,AX,AO,MAX,OUT,R,A,M,D,ACTION=action,LAZY=LAZY)
