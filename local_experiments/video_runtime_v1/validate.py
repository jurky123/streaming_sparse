"""Regression gates for cached readout, shared logits, sparse exact/tail and E49 scores."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from .scores import grouped_scores,candidate_pool,apply_best
from .page import encode,read
from .attention import Workspace,exact,compensate,merge
from .runtime import risk_from_logits
from .weights import load,ROOT
from ..review_v1.ops import Summary,RouteOps,Config
from ..review_v1.runtime import SelectedAttention
from ..e30_beta_diagnostics.run import local_scores
from ..e27_adaptive_query.core import page_features
from ..learned_cache.io import atomic

@torch.no_grad()
def run(dest):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(43)
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);fixtures=[]
    for path in sorted((ROOT/'artifacts/e49_numeric_v1').glob('reference_input*.npz')):
        z=np.load(path);rp,b,lm,c,mask,r,a=[torch.as_tensor(z[n],device='cuda') for n in ('rp','beta','logmass','center','mask','r','a')]
        r=r.long();a=a.long();nr=len(torch.unique_consecutive(r));na=len(r)//nr;rr=r.reshape(nr,na)[:,0];aa=a.reshape(nr,na)[0]
        old,_=local_scores(rp,b,lm,c,mask,r,a,1.);new=grouped_scores(rp[None],b[None],lm[None],c[None],mask[None],rr[None],aa[None])[0].flatten()
        error=float((old-new).abs().max());assert torch.isfinite(new).all() and torch.allclose(old,new,atol=1e-10,rtol=1e-8),(path,error)
        j=int(old.argmax());assert int(new.argmax())==j and bool(new[j]>0)==bool(old[j]>0),(path,float(old[j]),float(new[j]))
        fixtures.append(dict(file=path.name,event=str(z['event']),model=str(z['model']),max_abs=error,action=j,old_raw_gain=float(old[j]),new_raw_gain=float(new[j])))
    assert len(fixtures)>=14
    # Dominant remove regression: denominator must retain a background 1000 log units below it.
    rp=torch.randn(2,16,410,16,device='cuda',dtype=torch.float64)*.1;b=torch.randn(2,16,410,device='cuda',dtype=torch.float64)*.2;lm=torch.randn_like(b);center=torch.randn_like(rp)
    lm[:,:,0]+=1000;mask=(torch.arange(410,device='cuda')<41)[None].expand(2,-1)
    r=torch.arange(16,device='cuda')[None].expand(2,-1);a=torch.arange(41,73,device='cuda')[None].expand(2,-1)
    new=grouped_scores(rp,b,lm,center,mask,r,a)
    for i in range(2):
        old,_=local_scores(rp[i],b[i],lm[i],center[i],mask[i],r[i].repeat_interleave(32),a[i].repeat(16),1.)
        assert torch.isfinite(new[i]).all() and torch.allclose(old,new[i].flatten(),atol=1e-11,rtol=1e-9)
    model,reader,phi,B,identity=load();h=12;p=216;t=2;n=64*t
    ev=next((ROOT/'artifacts/full_coverage_v1/data/city_walk_s701').glob('c2_l15_s0_h0_t*.pt'))
    e=torch.load(ev,map_location='cpu',weights_only=True);kv=torch.load(ev.parent/e['kv_file'],map_location='cpu',weights_only=True)
    s=Summary(*(kv['summary'][name].cuda().expand(h,-1,*kv['summary'][name].shape[2:]).contiguous() for name in ('keys','mean','detail','stats')))
    q=torch.cat([e['q'],e['q']],0).cuda().bfloat16()[None].expand(h,-1,-1).contiguous()
    offsets=torch.linspace(0,63,16,device='cuda').round().long();probes=q.reshape(h,t,64,128).index_select(2,offsets)
    co,A,mean=encode(s,reader,phi,B.double());rp,bp=read(probes,co,A,phi);read_errors=[]
    for head in range(h):
        ss=s.slice(0,p);ss=Summary(*(getattr(ss,name)[head:head+1] for name in ('keys','mean','detail','stats')))
        ff=phi(probes[head].flatten(0,1).float(),ss);ro,bo=reader(ss,ff)
        read_errors.append(float((rp[head].flatten(0,1)-ro).abs().max()));assert torch.allclose(rp[head].flatten(0,1),ro,atol=2e-6,rtol=1e-5) and torch.allclose(bp[head].flatten(0,1),bo,atol=2e-6,rtol=1e-5)
    logits=torch.bmm(q.float(),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,64,p,4)/128**.5
    risk=risk_from_logits(probes,logits.index_select(2,offsets),s,model[15].theta)
    f,_=RouteOps.features(probes.flatten(1,2),s);ref_risk=(f*model[15].theta[:,None,None]).sum(-1).reshape(h,t,16,p).mean(2)
    own=torch.full((h,t),int(e['own']),device='cuda',dtype=torch.long)
    sel=RouteOps.select(risk,own,Config());oldsel=RouteOps.select(ref_risk,own,Config())
    overlap=float((sel['mask']&oldsel['mask']).sum()/oldsel['mask'].sum());assert overlap==1.
    count=int(p*.1);ids=sel['ids'][...,:count];mask=sel['mask']
    # Validate original stable pool including ties and no-op / budget / own invariants.
    rr,aa=candidate_pool(risk,mask,own)
    from ..e12_swap_prediction.core import pool
    for head in range(h):
        for tile in range(t):
            r0,a0=pool(dict(risk=risk[head,tile].cpu().numpy()),mask[head,tile].cpu().numpy(),int(own[head,tile]))
            assert np.array_equal(rr[head,tile].repeat_interleave(32).cpu(),r0) and np.array_equal(aa[head,tile].repeat(16).cpu(),a0)
    changed,gain,accepted=apply_best(mask.reshape(h*t,p),rr.reshape(h*t,16),aa.reshape(h*t,32),torch.zeros(h*t,16,32,device='cuda'))
    assert torch.equal(changed,mask.reshape(h*t,p)) and not accepted.any()
    rawk=kv['k'].cuda()[None].expand(h,-1,-1,-1).contiguous();rawv=kv['v'].cuda()[None].expand_as(rawk).contiguous()
    nativeq=q.transpose(0,1).contiguous()[None];nativek=rawk.flatten(1,2).transpose(0,1).contiguous()[None];nativev=rawv.flatten(1,2).transpose(0,1).contiguous()[None]
    w=Workspace();oe,le=exact(nativeq,nativek,nativev,ids,w);oa,la=compensate(logits,s,model[15].basis,mask);out=merge(oe,le,oa,la)[0].transpose(0,1)
    ref=SelectedAttention.fixed(q,rawk,rawv,s,model[15].basis,dict(ids=ids,mask=mask,fixed_count=count));delta=out.float()-ref.float()
    nmse=float(delta.square().sum()/ref.float().square().sum());max_abs=float(delta.abs().max());assert nmse<2e-5 and max_abs<.05,(nmse,max_abs)
    report=dict(passed=True,E49_fixtures=fixtures,dominant_remove_passed=True,cached_reader_max_abs=max(read_errors),risk_overlap=overlap,risk_relative_max_abs=float((risk-ref_risk).abs().max()/ref_risk.abs().max()),attention_nmse=nmse,attention_max_abs=max_abs,original_candidate_pool_exact=True,no_op_exact=True,identity=identity)
    atomic(dest/'validation.json',report);print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/video_runtime_v1/validation');a=p.parse_args();run(a.output)
