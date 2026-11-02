import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _shortlist(MASS,FEAT,CODE,W,MASK,ADDS,DROPS,AA,DD,NUM,DEN,ALL,OUT,CHANGES,P:tl.constexpr,STEPS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);h=row//72;i=tl.arange(0,16);r=tl.arange(0,8);p=tl.arange(0,B)
    ids=tl.where(i<8,tl.load(ADDS+row*8+i%8),tl.load(DROPS+row*8+i%8))
    avail=tl.where(i<8,tl.load(AA+row*8+i%8),tl.load(DD+row*8+i%8))
    mass=tl.load(MASS+row*P+ids);code=tl.load(CODE+(h*P+ids[:,None])*8+r[None,:]).to(tl.float32)
    feat=tl.load(FEAT+(row*P+ids[:,None])*8+r[None,:])
    num=tl.load(NUM+row*8+r);den=tl.load(DEN+row);allv=tl.load(ALL+row*8+r)
    mask=tl.load(MASK+row*P+p,p<P,0);count=tl.full((),0,tl.int32)
    # Native initial stage remains exact; subsequent stages are checked against reference.
    for iteration in range(STEPS-1):
        selected=num/den;missing=allv-selected;delta=code-selected[None,:]
        energy=tl.maximum(tl.sum(selected*selected,0),.01);align=tl.sum(delta*missing[None,:],1)/energy;mag=tl.sum(delta*delta,1)/energy
        weight=tl.load(W+h*14+r);score=tl.sum(feat*weight[None,:],1)
        score+= (i>=8).to(tl.float32)*tl.load(W+h*14+8)+tl.minimum(mass/den,2.)*tl.load(W+h*14+9)
        score+=(tl.minimum(tl.maximum(align,-10.),10.)/5.)*tl.load(W+h*14+10)+(tl.minimum(mag,10.)/5.)*tl.load(W+h*14+11)
        score+=tl.minimum(tl.maximum(mass*align,-2.),2.)*tl.load(W+h*14+12)+tl.minimum(mass*mag,2.)*tl.load(W+h*14+13)
        score=lib.log(tl.maximum(mass,1.e-12))+lib.tanh(score)
        high=tl.max(tl.where((i<8)&avail,score,-float('inf')),0);low=tl.min(tl.where((i>=8)&avail,score,float('inf')),0);change=(high>low)&(high<float('inf'))&(low>-float('inf'))
        add=tl.min(tl.where((i<8)&avail&(score==high),ids,2147483647),0);drop=tl.min(tl.where((i>=8)&avail&(score==low),ids,2147483647),0)
        ma=tl.sum(tl.where((i<8)&(ids==add),mass,0),0);md=tl.sum(tl.where((i>=8)&(ids==drop),mass,0),0)
        ca=tl.sum(tl.where(((i<8)&(ids==add))[:,None],code,0),0);cd=tl.sum(tl.where(((i>=8)&(ids==drop))[:,None],code,0),0)
        num+=tl.where(change,ma*ca-md*cd,0.);den=tl.maximum(den+tl.where(change,ma-md,0.),1.e-12)
        mask=tl.where(change&(p==drop),False,tl.where(change&(p==add),True,mask));avail=avail&~(change&((ids==add)|(ids==drop)));count+=change.to(tl.int32)
    tl.store(OUT+row*P+p,mask,p<P);tl.store(CHANGES+row,count)

def shortlist(mass,feat,code,w,mask,adds,drops,aa,dd,num,den,allv,steps):
    p=mass.shape[-1];out=torch.empty_like(mask);changes=torch.empty(mass.shape[:2],device=mass.device,dtype=torch.int32)
    _shortlist[(mass.shape[0]*mass.shape[1],)](mass,feat.contiguous(),code,w,mask,adds,drops,aa,dd,num,den,allv,out,changes,p,steps,tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False)
    return out,changes
