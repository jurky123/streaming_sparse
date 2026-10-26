import math
import torch
import triton as tr
import triton.language as tl
from ..opt8.projection import project
from ..opt17.route import _landmark,_coarse,_top
from ..opt19.route_fast import _part
@tr.jit
def _fine_age(Q,K,LC,IDS,S,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,PAGES:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(2);ci=tl.program_id(1)*8+tl.arange(0,8)
    pc=tl.load(IDS+(h*G+g)*C+ci,ci<C,0)
    age=tr.cdiv(B,PAGES)-1-pc//PAGES
    np=tl.where(age==0,16,tl.where(age==1,8,tl.where(age==2,4,2)))
    mp=tl.max(tl.where(ci<C,np,0),0)
    if mp<=2:_part(Q,K,LC,IDS,S,N,B,C,G,8,2)
    elif mp<=4:_part(Q,K,LC,IDS,S,N,B,C,G,8,4)
    elif mp<=8:_part(Q,K,LC,IDS,S,N,B,C,G,8,8)
    else:_part(Q,K,LC,IDS,S,N,B,C,G,8,16)

def scores(q,state,fraction,reps,seg,coarse_width=128):
    h,n,_=q.shape;b=state['key'].shape[1];g=tr.cdiv(n,32);c=math.ceil(b*fraction)
    qq=project(q,state['basis'])
    lm=torch.empty((h,g,reps,64),device=q.device,dtype=torch.float16)
    _landmark[(g,h)](qq,lm,n,g,reps,num_warps=4)
    coarse=torch.empty((h,g,b),device=q.device)
    off=0
    for kk,lc in state['parts']:
        np,p=lc.shape[1:];gp=coarse_width//p
        _coarse[(g,tr.cdiv(np,gp),h)](lm,kk,lc,coarse,n,b,g,off,np,p,gp,reps,num_warps=4,num_stages=1)
        off+=np
    ids=torch.empty((h,g,c),device=q.device,dtype=torch.int32)
    _top[(h*g,)](coarse,ids,h*g,b,c,tr.next_power_of_2(b),num_warps=4)
    score=torch.empty((h,n,c),device=q.device,dtype=torch.float32)
    _fine_age[(g,tr.cdiv(c,8),h)](qq,state['key'],state['logcount'],ids,score,n,b,c,g,tr.cdiv(seg,64),num_warps=4,num_stages=1)
    return score,ids
