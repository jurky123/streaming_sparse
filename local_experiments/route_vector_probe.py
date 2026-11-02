import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _features_vector(PROB,MASS,CODE,VV,KV,W,MASK,DEN,SEL,ALL,RATIO,STD,S,P:tl.constexpr,B:tl.constexpr):
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

    fi=tl.arange(0,16)
    values=tl.full((B,16),0.,tl.float32)
    values=tl.where(fi[None,:]==0,(tl.full((B,),1.,tl.float32))[:,None],values)
    values=tl.where(fi[None,:]==1,(lc)[:,None],values)
    values=tl.where(fi[None,:]==2,(peak)[:,None],values)
    values=tl.where(fi[None,:]==3,(spread)[:,None],values)
    values=tl.where(fi[None,:]==4,(vv)[:,None],values)
    values=tl.where(fi[None,:]==5,(kv)[:,None],values)
    values=tl.where(fi[None,:]==6,(current)[:,None],values)
    values=tl.where(fi[None,:]==7,(current*lc)[:,None],values)
    values=tl.where(fi[None,:]==8,(mask)[:,None],values)
    values=tl.where(fi[None,:]==9,(tl.load(RATIO+row*P+p,p<P,0))[:,None],values)
    values=tl.where(fi[None,:]==10,(tl.minimum(tl.maximum(align,-10.),10.)/5.)[:,None],values)
    values=tl.where(fi[None,:]==11,(tl.minimum(mag,10.)/5.)[:,None],values)
    values=tl.where(fi[None,:]==12,(tl.minimum(tl.maximum(mass*align,-2.),2.))[:,None],values)
    values=tl.where(fi[None,:]==13,(tl.minimum(mass*mag,2.))[:,None],values)
    tl.store(S+(row*P+p[:,None])*14+fi[None,:],values,(p[:,None]<P)&(fi[None,:]<14))

def main():
    import statistics,json
    from pathlib import Path
    from .route_exact_features import _features_exact
    from .learned_cache.io import atomic
    torch.set_num_threads(4)
    e={k:v.cuda() for k,v in torch.load('artifacts/route_kernel_20261031/debug/boundary.pt',weights_only=True).items()};p=410
    for k in ['prob','mass','mask','den','sel','allvalue','sf','score']:
        e[k]=e[k].expand(12,72,*e[k].shape[2:]).contiguous()
    for k in ['code','vv','kv','w']:e[k]=e[k].expand(12,*e[k].shape[1:]).contiguous()
    spread=e['prob'].clamp_min(1e-12).log().std(2,unbiased=False).clamp(0,5)/5;ratio=(e['mass']/e['den']).clamp(0,2);outs=[torch.empty_like(e['sf']),torch.empty_like(e['sf'])]
    args=[e[k] for k in ['prob','mass','code','vv','kv','w','mask','den','sel','allvalue']]+[ratio,spread]
    def feature(kernel,out):kernel[(864,)](*args,out,p,tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False);return out
    feature(_features_exact,outs[0]);feature(_features_vector,outs[1]);assert torch.equal(*outs)
    rows=[]
    for name,fn in [('features_strided',lambda:feature(_features_exact,outs[0])),('features_vector',lambda:feature(_features_vector,outs[1])),('native_std',lambda:e['prob'].clamp_min(1e-12).log().std(2,unbiased=False).clamp(0,5)/5),('native_score',lambda:e['mass'].clamp_min(1e-12).log()+torch.tanh(torch.einsum('htpf,hf->htp',outs[0],e['w'])))]:
        for _ in range(3):fn()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):out=fn()
        times=[]
        for _ in range(5):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(100):g.replay()
            b.record();b.synchronize();times.append(a.elapsed_time(b)/100)
        rows.append(dict(name=name,ms=statistics.median(times)))
    atomic(Path('artifacts/route_kernel_20261031/exact_primitives.json'),dict(features_bit_exact=True,rows=rows));print(json.dumps(rows))
if __name__=='__main__':main()
