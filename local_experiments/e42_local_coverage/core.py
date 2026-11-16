"""E42: fixed 50/50 true-mass supplementation, unchanged online model."""
import argparse,collections,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from ..e41_training_signal.run import (ROOT,OUT,E29,CACHE,LocalReader,bundle_for,Summary,contributions,summary_output,route,gains,objective,rms_floor,reconstruct,safe_predictions,local_scores,table,dump,sha,losses,parameter_audit)
from ..e29_local_reader.core import load_phi,checkpoint,E27
from ..e39_action_margin.core import original_set_loss
D=ROOT/'artifacts/e42_local_coverage_v1';E39=ROOT/'artifacts/e39_action_margin_v1';E41=ROOT/'artifacts/e41_training_signal_v1'
def protocol():
 D.mkdir(exist_ok=True)
 if (D/'protocol.json').exists():return
 selected={}
 for seed in [17,29]:
  meta=json.loads((CACHE/f'train_s{seed}/metadata.json').read_text());groups=collections.defaultdict(list)
  for m in meta:
   if m['head']==9:groups[m['trajectory']].append(m)
  ss=[]
  for i,(name,rs) in enumerate(sorted(groups.items())):
   rs=sorted(rs,key=lambda x:x['index']);ss.append(rs[(i*7)%len(rs)]['event'])
  selected[str(seed)]=ss
 worst=['data/forest_run_s702/c8_l29_s0_h9_t39.pt','data/reef_fish_s703/c5_l29_s0_h9_t37.pt']
 dump(D/'protocol.json',dict(experiment='E42',seeds=[17,29],mixture=.5,weight='0.5*m^2+0.5*sum(m^2)*alpha_true^2/sum(alpha_true^2), separately per Query; detached',local_scale='unchanged E29 train-only scale',unchanged=['C','phi','B16','Reader 51157','beta loss','gain/listwise set loss','Adam reset lr1e-4 betas(.9,.999) eps1e-8','global norm clip1','M16','Risk','Medium512','count/own','original E39 state schedule'],train_updates=2304,checkpoint='final epoch1; epoch0 saved only',baseline='reuse E39-T0 only after provenance and 16-step numerical prefix replay',prefix='16 disposable T0 updates per seed for contract verification, not a new baseline',head9_events=selected,worst_events=worst,preflight_gate='Both seeds: event-mean old-underestimated subset shares of local loss and gradient L2 increase for all/harmful/oracle endpoint roles; weight conservation and finite gradients. No coefficient adjustment.',mechanism_gate='Both seeds T1 vs T0: train net gain up and H_loss down, dev net gain up; record tail and trajectories separately',deployment_gate='both seed recoveries >=20%; both dev trajectories positive; Set>Independent and no-op; mean relative gain nonnegative; harm count, P10, worst absolute harm improve vs E29',test='sealed regardless of diagnostic progress; no automatic unsealing',raw_KV=False,teacher_input=False,mix_search=False))

class Data:
 def __init__(self,seed):
  self.seed=seed;self.bundle=bundle_for('C',seed);self.phi=load_phi(seed)
  for p in list(self.bundle.parameters())+list(self.phi.parameters()):p.requires_grad_(False)
  self.fit=torch.load(OUT/'fit.pt',weights_only=True);self.B=self.fit['B'].cuda().double();self.scale=json.loads((E29/f's{seed}/local_scale.json').read_text())['scale'];self.qs=torch.linspace(0,63,16,device='cuda').round().long();self.meta={};self.summaries={}
  for split in ['train','development']:
   ms=json.loads((CACHE/f'{split}_s{seed}/metadata.json').read_text())
   for m in ms:self.meta[m['event']]=m
  self.frozen={k:v.detach().cpu().clone() for k,v in self.bundle.layers.state_dict().items()};self.fphi={k:v.detach().cpu().clone() for k,v in self.phi.state_dict().items()}
 @torch.no_grad()
 def load(self,event,state=None):
  meta=self.meta[event];cd=CACHE/f'{meta["split"]}_s{self.seed}';ev=torch.load(OUT/event,map_location='cpu',weights_only=True);assert ev['split'] in ['train','development'];key='l%sh%s'%(ev['layer'],ev['head']);sid=str(cd/meta['summary_file'])
  # Keep CPU summaries only; fixed-size GPU event buffers are released between calls.
  if sid not in self.summaries:self.summaries[sid]=torch.load(sid,map_location='cpu',weights_only=True)['summary']
  s=Summary(*(self.summaries[sid][k].cuda() for k in ['keys','mean','detail','stats']))
  q=ev['q'].cuda().float();N=ev['N'].cuda().double();Z=ev['Z'].cuda().double();dense=ev['dense'].cuda().double();logscale=ev['log_scale'].cuda().double();layer=self.bundle.layers[key];c=contributions(q,s,layer.basis,logscale);lm=c['logzh']-c['A'].log()[:,None];bt=ev['page_logz'].cuda().double()-logscale[:,None]-c['logzh'];ap=summary_output(c,s,layer.basis);rt=(ev['page_output'].cuda().double()-ap)@self.B;cen=(ap-c['o'][:,None])@self.B;F=self.phi(q,s);qs=self.qs
  if state is None:mask,r,a=route(q[qs],s,layer,ev['own'])
  else:mask=torch.tensor(state['mask'],device='cuda',dtype=torch.bool);r=torch.tensor(state['remove'],device='cuda');a=torch.tensor(state['add'],device='cuda')
  assert bool(mask[ev['own']]) and bool(mask[r].all()) and bool((~mask[a]).all()) and not bool((r==ev['own']).any())
  g=gains(c,N,Z,dense,mask,r,a);m=c['mass'][qs];alpha=ev['page_logz'].cuda().double()[qs].softmax(-1);w0=m.square();w1=.5*w0+.5*w0.sum(-1,keepdim=True)*alpha.square()/alpha.square().sum(-1,keepdim=True)
  assert not w1.requires_grad and torch.allclose(w1.sum(-1),w0.sum(-1),rtol=1e-12,atol=1e-15) and bool((w1>=.5*w0).all())
  return dict(meta=meta,s=s,F=F[qs],rtrue=rt[qs],btrue=bt[qs],mass=m,lm=lm[qs],center=cen[qs],mask=mask,r=r,a=a,g=g,norm=dense.square().sum()/64+1e-12,initial=float(objective(c,N,Z,dense,mask)),w0=w0,w1=w1,alpha=alpha,own=ev['own'])
 def check_frozen(self):
  assert all(torch.equal(v.cpu(),self.frozen[k]) for k,v in self.bundle.layers.state_dict().items());assert all(torch.equal(v.cpu(),self.fphi[k]) for k,v in self.phi.state_dict().items())

def model_from(path):
 m=LocalReader().cuda();x=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(x.get('model',x));return m

def loss_terms(model,d,data,new):
 rp,beta=model(d['s'],d['F']);bs=rms_floor(d['btrue'],(0,),data.fit['bscale']);gs=rms_floor(d['g'],(0,),data.fit['gscale']).reshape(())
 if new:local=(d['w1'][...,None]*(rp-d['rtrue']).square()).mean()/data.scale**2
 else:local=((d['mass'][...,None]*(rp-d['rtrue']))/data.scale).square().mean()
 quality=torch.nn.functional.smooth_l1_loss(beta/bs,d['btrue']/bs)
 pg,_=safe_predictions(reconstruct(rp,beta,d['lm'],d['center']),beta,d['lm'],d['mask'],d['r'],d['a'],d['norm']);sl=original_set_loss(pg,d['g'],gs)
 return [local,quality,sl],rp,beta,gs

def contract(seed,data):
 rec=json.loads((E39/f'T0_s{seed}/training_complete.json').read_text());sch=E39/f'schedule_s{seed}.json';schedule=json.loads(sch.read_text());assert len(schedule)==2304 and rec['updates']==2304 and rec['parameters']==51157
 assert rec['schedule_sha']==sha(sch) and rec['initial_sha']==sha(E29/f's{seed}/selected.pt') and rec['C_sha']==sha(checkpoint('C',seed)) and rec['phi_sha']==sha(E27/f's{seed}/selected.pt')
 for epoch in range(2):
  seg=schedule[epoch*1152:(epoch+1)*1152];assert len(set(s['event'] for s in seg))==1152 and all(data.meta[s['event']]['split']=='train' and s['mode']==s['update']%10 for s in seg)
 m=model_from(E29/f's{seed}/selected.pt');original=torch.load(E39/f'T0_s{seed}/initial.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v.cpu(),original[k]) for k,v in m.state_dict().items())
 opt=torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);assert len(opt.state)==0
 old=json.loads((E39/f'T0_s{seed}/curves.json').read_text());checks=[]
 for st in schedule[:16]:
  d=data.load(st['event'],st);ls,_,_,gs=loss_terms(m,d,data,False);loss=sum(ls);opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();o=old[st['update']];values=dict(local_loss=float(ls[0]),quality_loss=float(ls[1]),set_loss=float(ls[2]),loss=float(loss),gradnorm=float(gn),gain_scale=float(gs));errs={k:abs(v-o[k]) for k,v in values.items()};assert all(np.isclose(v,o[k],rtol=2e-5,atol=2e-7) for k,v in values.items()),(seed,st['update'],values,o)
  checks.append(dict(update=st['update'],event=st['event'],**{k+'_error':v for k,v in errs.items()}))
 data.check_frozen();table(D/f'contract_prefix_s{seed}.csv',checks);dump(D/f'contract_s{seed}.json',dict(passed=True,baseline_reused=True,prefix_updates_discarded=16,source_code_sha=sha(ROOT/'local_experiments/e39_action_margin/core.py'),T0_sha=sha(E39/f'T0_s{seed}/epoch1.pt'),schedule_sha=sha(sch),initial_sha=rec['initial_sha'],C_sha=rec['C_sha'],phi_sha=rec['phi_sha'],prefix_max_error=max(max(r[k] for k in r if k.endswith('_error')) for r in checks),raw_KV=False,test=False))
 print('CONTRACT PASS',seed,flush=True)


def train(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);protocol();assert json.loads((D/'preflight_gate.json').read_text())['passed'];assert json.loads((D/f'contract_s{seed}.json').read_text())['passed'];dest=D/f'T1_s{seed}';dest.mkdir(exist_ok=True);assert not (dest/'training_complete.json').exists();data=Data(seed);model=model_from(E29/f's{seed}/selected.pt');torch.save(model.state_dict(),dest/'initial.pt');opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);assert not opt.state
 schedule=json.loads((E39/f'schedule_s{seed}.json').read_text());curves=[];start=time.time()
 for st in schedule:
  d=data.load(st['event'],st);ls,rp,bp,gs=loss_terms(model,d,data,True);loss=sum(ls);assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();u=st['update'];curves.append(dict(update=u,epoch=u//1152,event=st['event'],local_loss=float(ls[0]),quality_loss=float(ls[1]),set_loss=float(ls[2]),loss=float(loss),gradnorm=float(gn),clip_scale=min(1.,1./(float(gn)+1e-6)),gain_scale=float(gs)))
  if u%192==0:print('TRAIN',seed,u,round(time.time()-start),flush=True);dump(dest/'status.json',dict(update=u,seconds=time.time()-start))
  if (u+1)%1152==0:
   epoch=u//1152;torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=u+1),dest/f'epoch{epoch}.pt');dump(dest/'curves.json',curves)
 data.check_frozen();dump(dest/'training_complete.json',dict(seed=seed,updates=len(curves),final_epoch=1,parameters=sum(p.numel() for p in model.parameters()),initial_sha=sha(E29/f's{seed}/selected.pt'),schedule_sha=sha(E39/f'schedule_s{seed}.json'),C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}/selected.pt'),frozen_C=True,frozen_phi=True,beta_loss_unchanged=True,set_loss_unchanged=True,raw_KV=False,test=False,seconds=time.time()-start));print('TRAIN DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();train(a.seed)
