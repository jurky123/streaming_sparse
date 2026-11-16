"""E41 frozen E29 audit. No optimizer, raw KV reads, or Test access."""
import argparse, collections, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from ..e29_local_reader.core import (ROOT,OUT,E29,LocalReader,bundle_for,Summary,contributions,summary_output,route,gains,objective,rms_floor,reconstruct,safe_predictions)
from ..e30_beta_diagnostics.run import local_scores
D=ROOT/'artifacts/e41_training_signal_v1'; CACHE=ROOT/'artifacts/e40_reader_scale_v1'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 if not rows:return
 with p.open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protocol():
 D.mkdir(exist_ok=True)
 if (D/'protocol.json').exists():return
 selected={}
 for seed in [17,29]:
  metas=json.loads((CACHE/f'train_s{seed}/metadata.json').read_text()); groups=collections.defaultdict(list)
  for m in metas:groups[(m['trajectory'],m['layer'])].append(m)
  out=[]
  for k,v in sorted(groups.items()):
   v=sorted(v,key=lambda m:(m['chunk'],m['step'],m['head'],m['tile']))
   # Deterministic strata, before inspecting losses or gradients.
   for i in [len(v)//4,3*len(v)//4]:out.append(v[i]['event'])
  assert len(out)==48;selected[str(seed)]=out
 dump(D/'protocol.json',dict(experiment='E41',training=False,test=False,raw_KV=False,updates=0,seeds=[17,29],splits=['train','development'],full_events_per_seed=1440,parameter_gradient_events=selected,parameter_states='Risk only; frozen final E29 snapshot, not historical trajectory',loss='Exact E29 local+quality+set, original scales and reductions; no changes',clip='global norm 1; common scalar preserves within-event angles and relative norms',reference='KL(true mass || predicted mass), mean over queries; diagnostic only',roles='endpoint union / other candidates / background partition; additional overlapping roles separately',selection='48 events per seed: trajectory x layer, sorted chunk/step/head/tile at quartile indices',stop='No automatic probe, training, weight sweep, or Test evaluation'))

def losses(rp,bp,rt,bt,m,lm,center,mask,r,a,g,norm,scale,fit):
 bs=rms_floor(bt,(0,),fit['bscale']);gs=rms_floor(g,(0,),fit['gscale']).reshape(())
 lc=((m[...,None]*(rp-rt))/scale).square().mean()
 lb=torch.nn.functional.smooth_l1_loss(bp/bs,bt/bs)
 sc,ind=safe_predictions(reconstruct(rp,bp,lm,center),bp,lm,mask,r,a,norm)
 target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0)
 rank=-(target*torch.log_softmax(torch.cat([sc,sc.new_zeros(1)])/gs,0)).sum()
 ls=torch.nn.functional.smooth_l1_loss(sc/gs,g/gs)+.1*rank
 return [lc,lb,ls],bs,gs,sc,ind

def parameter_audit(model,ls,meta):
 pars=list(model.parameters());names=list(dict(model.named_parameters())); grads=[]
 for loss in ls:
  gg=torch.autograd.grad(loss,pars,retain_graph=True,allow_unused=True)
  grads.append([torch.zeros_like(p) if v is None else v.detach() for p,v in zip(pars,gg)])
 vectors={}
 beta_rows=torch.arange(16,85,17,device='cuda');other=torch.tensor([i for i in range(85) if i%17!=16],device='cuda')
 for group in ['all','hidden','beta_output','response_output']:
  vv=[]
  for gg in grads:
   bits=[]
   for n,v in zip(names,gg):
    if group=='all' or (group=='hidden' and n.startswith('page.0')):bits.append(v.flatten())
    elif n.startswith('page.2') and group in ['beta_output','response_output']:bits.append(v[beta_rows if group=='beta_output' else other].flatten())
   vv.append(torch.cat(bits).double())
  vectors[group]=vv
 total=sum(vectors['all']); norm=float(total.norm());clip=min(1.,1./(norm+1e-6));rows=[]
 for group,vv in vectors.items():
  t=sum(vv);ns=[float(v.norm()) for v in vv];tn=float(t.norm());row=dict(**meta,parameter_group=group,total_norm=tn,global_norm=norm,clip_scale=clip,cancellation_ratio=tn/(sum(ns)+1e-300))
  for i,name in enumerate(['local','beta','set']):
   row[name+'_norm']=ns[i];row[name+'_clipped_norm']=ns[i]*clip
   row[name+'_total_projection']=float(vv[i]@t)/(tn*tn) if tn else None
  for i,j in [(0,1),(0,2),(1,2)]:row['cos_'+['local','beta','set'][i]+'_'+['local','beta','set'][j]]=float(vv[i]@vv[j])/(ns[i]*ns[j]) if ns[i]*ns[j] else None
  rows.append(row)
 return rows

def run(seed,limit=0):
 torch.set_num_threads(2);protocol();dest=D/(f'preflight_s{seed}' if limit else f's{seed}');dest.mkdir(exist_ok=True)
 assert not (dest/'complete.json').exists()
 model=LocalReader().cuda().eval();mp=E29/f's{seed}/selected.pt';model.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['model']);frozen={k:v.detach().clone() for k,v in model.state_dict().items()};bundle=bundle_for('C',seed)
 for p in bundle.parameters():p.requires_grad_(False)
 fit=torch.load(OUT/'fit.pt',weights_only=True);B=fit['B'].cuda().double();scale=json.loads((E29/f's{seed}/local_scale.json').read_text())['scale'];qs=torch.linspace(0,63,16,device='cuda').round().long();selected=set(json.loads((D/'protocol.json').read_text())['parameter_gradient_events'][str(seed)])
 decisions=[];roles=[];gradrows=[];checks=[];sources=[];start=time.time()
 for split in ['train','development']:
  cd=CACHE/f'{split}_s{seed}';metas=json.loads((cd/'metadata.json').read_text());metas=metas[:limit] if limit else metas
  for ix,meta in enumerate(metas):
   ep=OUT/meta['event'];ev=torch.load(ep,map_location='cpu',weights_only=True);assert ev['split']==split
   sp=cd/meta['summary_file'];sx=torch.load(sp,map_location='cpu',weights_only=True);s=Summary(*(sx['summary'][k].cuda() for k in ['keys','mean','detail','stats']));cache=np.load(cd/('event_%04d.npz'%meta['index']))
   q=ev['q'].cuda().float();key='l%sh%s'%(ev['layer'],ev['head']);N=ev['N'].cuda().double();Z=ev['Z'].cuda().double();dense=ev['dense'].cuda().double();logscale=ev['log_scale'].cuda().double()
   with torch.no_grad():
    c=contributions(q,s,bundle.layers[key].basis,logscale);lm=c['logzh']-c['A'].log()[:,None];bt=ev['page_logz'].cuda().double()-logscale[:,None]-c['logzh'];ap=summary_output(c,s,bundle.layers[key].basis);rt=(ev['page_output'].cuda().double()-ap)@B;center=(ap-c['o'][:,None])@B
    ff=torch.as_tensor(cache['F'],device='cuda').double();F=torch.cat([torch.ones_like(ff[...,:1]),ff],-1);mask,r,a=route(q[qs],s,bundle.layers[key],ev['own']);g=gains(c,N,Z,dense,mask,r,a);J=float(objective(c,N,Z,dense,mask));P=mask.numel();m=c['mass'][qs];l=lm[qs];rt=rt[qs];bt=bt[qs];center=center[qs];norm=dense.square().sum()/64+1e-12
   rp,bp=model(s,F[qs]);cached_beta=torch.as_tensor(cache['predicted_beta'][qs.cpu().numpy()],device='cuda');replay_error=float((bp.detach()-cached_beta).abs().max());assert torch.allclose(bp.detach(),cached_beta,rtol=2e-4,atol=2e-4),(replay_error,meta['event'])
   args=(rt,bt,m,l,center,mask,r,a,g,norm,scale,fit)
   ls,bs,gs,sc,ind=losses(rp,bp,*args)
   base={k:meta[k] for k in ['seed','event','split','trajectory','layer','head','chunk','step','tile']}
   jo=int(g.argmax()) if float(g.max())>0 else -1;j=int(sc.argmax());jp=j if float(sc[j])>0 else -1
   assert jp==meta['j_pred'] and jo==meta['j_oracle'],(base,jp,meta['j_pred'])
   for policy,score in [('Set',sc),('Independent',ind)]:
    jj=int(score.argmax());ex=float(score[jj])>0;gain=float(g[jj]);decisions.append(dict(**base,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),proposal_gain=gain,hindsight_gain=max(0.,gain),executed_gain=gain if ex else 0.,execute=ex,proposal_index=jj,predicted_gain=float(score[jj]),local_loss=float(ls[0]),beta_loss=float(ls[1]),set_loss=float(ls[2]),gain_scale=float(gs)))
   if meta['event'] in selected or limit:gradrows.extend(parameter_audit(model,ls,base))
   # Separate leaf outputs isolate loss allocation from shared encoder Jacobian.
   rr=rp.detach().requires_grad_();bb=bp.detach().requires_grad_();loss,bs,gs,_,_=losses(rr,bb,*args)
   lr=torch.autograd.grad(loss[0],rr,retain_graph=True)[0];gb=torch.autograd.grad(loss[1],bb,retain_graph=True)[0];sr,sb=torch.autograd.grad(loss[2],(rr,bb))
   expected_b=((bb.double()-bt)/bs).clamp(-1,1)/bs/bb.numel();expected_r=2*m[...,None].square()*(rr.double()-rt)/scale**2/rr.numel()
   be=float((gb.double()-expected_b).abs().max());re=float((lr.double()-expected_r).abs().max())
   assert torch.allclose(gb.double(),expected_b,rtol=3e-5,atol=1e-10) and torch.allclose(lr.double(),expected_r,rtol=3e-5,atol=1e-10)
   with torch.no_grad():
    alpha=(l+bb).softmax(-1);truth=(l+bt).softmax(-1);kl=(alpha-truth)/16
    lc=((m[...,None]*(rr-rt))/scale).square().sum(-1)/rr.numel();lb=torch.nn.functional.smooth_l1_loss(bb/bs,bt/bs,reduction='none')/bb.numel();err=bb-bt
    co=sx['coef'].cuda().double();low=co[:,0,16]-co[:,1:,16].abs().sum(-1);high=co[:,0,16]+co[:,1:,16].abs().sum(-1);outside=(low-bt).clamp_min(0)+(bt-high).clamp_min(0)
    C=torch.zeros(P,dtype=torch.bool,device='cuda');C[r]=True;C[a]=True;pred=torch.zeros_like(C);opt=pred.clone()
    if jp>=0:pred[r[jp]]=True;pred[a[jp]]=True
    if jo>=0:opt[r[jo]]=True;opt[a[jo]]=True
    end=pred|opt;flags={'all':torch.ones_like(C),'endpoint_pair':end,'other_candidate':C&~end,'background':~C,'predicted_endpoint':pred,'oracle_endpoint':opt,'harmful_predicted_endpoint':pred&(float(g[jp])< -1e-12 if jp>=0 else False)}
    assert torch.all(end.to(torch.int)+ (C&~end).to(torch.int)+(~C).to(torch.int)==1)
    metrics={'summary_mass':m,'predicted_mass':alpha,'true_mass':truth,'mass2':m.square(),'local_loss':lc,'beta_loss':lb,'local_grad_l1':lr.abs().sum(-1),'beta_grad_l1':gb.abs(),'set_beta_grad_l1':sb.abs(),'set_response_grad_l1':sr.abs().sum(-1),'kl_grad_l1':kl.abs(),'beta_correction_work':gb.double()*err,'set_beta_correction_work':sb.double()*err,'kl_correction_work':kl*err,'set_response_correction_work':(sr.double()*(rr-rt)).sum(-1)}
    for role,sel in flags.items():
     if not bool(sel.any()):continue
     row=dict(**base,role=role,pages=int(sel.sum()),page_fraction=float(sel.double().mean()),beta_mae=float(err[:,sel].abs().mean()),beta_bias=float(err[:,sel].mean()),beta_interval_outside_fraction=float((outside[:,sel]>0).double().mean()),interval_excess_mean=float(outside[:,sel].mean()),beta_scale_mean=float(bs[:,sel].mean()),under_summary_true_mass_fraction=float((truth[:,sel]*(m[:,sel]<truth[:,sel]/10)).sum()/truth.sum()),under_predicted_true_mass_fraction=float((truth[:,sel]*(alpha[:,sel]<truth[:,sel]/10)).sum()/truth.sum()),beta_gradient_correct_fraction=float((gb[:,sel]*err[:,sel]>0).double().mean()),set_beta_gradient_correct_fraction=float((sb[:,sel]*err[:,sel]>0).double().mean()))
     for name,val in metrics.items():
      sm=float(val[:,sel].sum());total=float(val.sum());row[name+'_sum']=sm;row[name+'_share']=sm/total if abs(total)>1e-300 else None
     roles.append(row)
   checks.append(dict(**base,beta_gradient_max_error=be,local_gradient_max_error=re,cache_beta_replay_error=replay_error,loss_finite=all(bool(torch.isfinite(v)) for v in loss)))
   sources.append(dict(event=meta['event'],split=split,event_sha=sha(ep),summary_file=meta['summary_file'],summary_sha=sha(sp)))
   if ix%48==0:
    print('AUDIT',seed,split,ix,'seconds',round(time.time()-start),flush=True)
    dump(dest/'status.json',dict(split=split,index=ix,seconds=time.time()-start))
   del rp,bp,ls,rr,bb,loss
  table(dest/'decisions.csv',decisions);table(dest/'output_gradient_roles.csv',roles);table(dest/'parameter_gradients.csv',gradrows);table(dest/'checks.csv',checks)
 assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items())
 dump(dest/'sources.json',sources);dump(dest/'complete.json',dict(seed=seed,events=len(checks),parameter_events=len(gradrows)//4,model_sha=sha(mp),weights_unchanged=True,training=False,optimizer_created=False,raw_KV_read=False,test=False,seconds=time.time()-start));print('DONE',seed,len(checks),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();run(a.seed,a.limit)
