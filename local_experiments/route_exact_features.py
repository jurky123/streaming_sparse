import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _features_exact(PROB,MASS,CODE,VV,KV,W,MASK,DEN,SEL,ALL,RATIO,STD,S,P:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);h=row//72;t=row%72;p=tl.arange(0,B);d=tl.arange(0,8)
    mass=tl.load(MASS+row*P+p,p<P,0);lm=lib.log(tl.maximum(mass,1.e-12));lc=tl.minimum(tl.maximum(lm,-20.),0.)/5.
    a=tl.load(PROB+((h*72+t)*4+0)*P+p,p<P,0);b=tl.load(PROB+((h*72+t)*4+1)*P+p,p<P,0)
    c=tl.load(PROB+((h*72+t)*4+2)*P+p,p<P,0);e=tl.load(PROB+((h*72+t)*4+3)*P+p,p<P,0)
    spread=tl.load(STD+row*P+p,p<P,0)
    peak=tl.minimum(tl.maximum(lib.log(tl.maximum(tl.maximum(tl.maximum(a,b),tl.maximum(c,e)),1.e-12))-lm,0.),10.)/5.
    vv=lib.log1p(tl.maximum(tl.load(VV+h*P+p,p<P,0),0.));kv=tl.load(KV+h*P+p,p<P,0)
    current=(p>=P-72).to(tl.float32);mask=tl.load(MASK+row*P+p,p<P,0).to(tl.float32);den=tl.load(DEN+row)
    sel=tl.load(SEL+row*8+d);missing=tl.load(ALL+row*8+d)-sel
    code=tl.load(CODE+(h*P+p[:,None])*8+d[None,:],p[:,None]<P,0).to(tl.float32);diff=code-sel[None,:]
    energy=tl.maximum(tl.sum(tl.sum(tl.reshape(tl.sum(tl.reshape(sel*sel,(4,2)),1),(2,2)),1),0),.01)
    ax=diff*missing[None,:];mx=diff*diff
    asum=tl.sum(tl.sum(tl.reshape(tl.sum(tl.reshape(ax,(B,4,2)),2),(B,2,2)),2),1)
    msum=tl.sum(tl.sum(tl.reshape(tl.sum(tl.reshape(mx,(B,4,2)),2),(B,2,2)),2),1)
    align=tl.div_rn(asum,energy);mag=tl.div_rn(msum,energy)
    tl.store(S+(row*P+p)*14+0,tl.full((B,),1.,tl.float32),p<P)
    tl.store(S+(row*P+p)*14+1,lc,p<P)
    tl.store(S+(row*P+p)*14+2,peak,p<P)
    tl.store(S+(row*P+p)*14+3,spread,p<P)
    tl.store(S+(row*P+p)*14+4,vv,p<P)
    tl.store(S+(row*P+p)*14+5,kv,p<P)
    tl.store(S+(row*P+p)*14+6,current,p<P)
    tl.store(S+(row*P+p)*14+7,current*lc,p<P)
    tl.store(S+(row*P+p)*14+8,mask,p<P)
    tl.store(S+(row*P+p)*14+9,tl.load(RATIO+row*P+p,p<P,0),p<P)
    tl.store(S+(row*P+p)*14+10,tl.minimum(tl.maximum(align,-10.),10.)/5.,p<P)
    tl.store(S+(row*P+p)*14+11,tl.minimum(mag,10.)/5.,p<P)
    tl.store(S+(row*P+p)*14+12,tl.minimum(tl.maximum(mass*align,-2.),2.),p<P)
    tl.store(S+(row*P+p)*14+13,tl.minimum(mass*mag,2.),p<P)

def main():
    e={k:v.cuda() for k,v in torch.load('artifacts/route_kernel_20261031/debug/boundary.pt',weights_only=True).items()};p=e['mass'].shape[-1];out=torch.empty_like(e['sf'])
    spread=e['prob'].clamp_min(1e-12).log().std(2,unbiased=False).clamp(0,5)/5
    _features_exact[(1,)](e['prob'],e['mass'],e['code'],e['vv'],e['kv'],e['w'],e['mask'],e['den'],e['sel'],e['allvalue'],(e['mass']/e['den']).clamp(0,2),spread,out,p,tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False)
    for i in range(14):print(i,'count',int((out[...,i]!=e['sf'][...,i]).sum()),'max',float((out[...,i]-e['sf'][...,i]).abs().max()),flush=True)
    score=e['mass'].log()+torch.tanh(torch.einsum('htpf,hf->htp',out,e['w']))
    print('scores different',int((score!=e['score']).sum()),'max',float((score-e['score']).abs().max()))
    for page in [379,402]:print(page,float(e['score'][0,0,page]),float(score[0,0,page]))
if __name__=='__main__':main()
