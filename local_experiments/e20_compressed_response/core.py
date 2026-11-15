import json,time,csv,math,hashlib
from pathlib import Path
import numpy as np
import torch
from ..e19_response_moments.core import ROOT,DATA,E13,sha,dump
from ..e19_response_moments.moments import gain_scores,array
from ..e17_swap_decomposition.evaluate import metrics
OUT=ROOT/'artifacts/e20_compressed_response_v1';E19=ROOT/'artifacts/e19_response_moments_v1'
EVENTS=['city_walk_s101_c5_l15_s0_h8_t61','horse_field_s101_c5_l15_s0_h8_t61']
CHECKPOINT=ROOT/'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'
CONFIGS=[('K4_PCA16',4,'pca16'),('K8_current8',8,'current8'),('K8_PCA8',8,'pca8'),('K8_PCA16',8,'pca16')]
def table(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def setup():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():return
 state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model'];f=np.load(DATA/'contributions'/EVENTS[0]/'summary_and_route_features.npz');assert np.array_equal(state['layers.15.basis'][8].bfloat16().float().numpy(),f['basis'])
 uk=np.linalg.qr(state['layers.15.bank'][8].double().numpy().T)[0];uv=np.linalg.qr(f['basis'].astype(float))[0];pca=np.load(E19/'pca16.npz')['V'];np.savez(OUT/'bases.npz',key4=uk[:,:4].astype('float32'),key8=uk[:,:8].astype('float32'),current8=uv.astype('float32'),pca8=pca[:,:8].astype('float32'),pca16=pca.astype('float32'))
 dump(OUT/'protocol.json',dict(created=time.time(),scope='E20 only TWO existing snapshots, same layer15/head8/chunk5/step0, no fitting,newvideo,QKVcapture,production code',basis='UK = QR of existing frozen layer15/head8 32-direction Key-response bank, first4/8 columns, no snapshot PCA. UV=current per-head rank8 QR OR frozen E19 V-response PCA first8/16. Shared across pages, stored FP32. No invented current rank16.',basis_sha=sha(OUT/'bases.npz'),checkpoint_sha=sha(CHECKPOINT),E19_PCA_sha=sha(E19/'pca16.npz'),stage1='original M2 vs mean-centered M2; mu=mean(delta), eta=delta-mu; exact exp(mu)*(1+eta+eta^2/2). Choose on city ONLY by actual M16 full-legal one-swap gain, tie within1e-10 by lower Risk-unselected net residual NMSE; ties original. Freeze choice before compression, horse not used to select.',compression='K-only r4/8 with fullV and V-only current8/PCA8/PCA16 with fullK offline FP64. Four joint configs K4/PCA16,K8/current8,K8/PCA8,K8/PCA16. Each full128 a0; original or centered moments consistently; symmetric Key second moment upper-triangle stores offdiagonal once with factor2 at contraction.',precision='FP64-statistics controls vs PRIMARY actual BF16 stored b1,B2,a0,A1,A2 and FP32 shared bases, evaluated in FP64 for numerical diagnostics. No silent clamping of negative masses or denominators.',streaming='all-page statistics vs history-only: keep original Risk and full legal actions, set CURRENT additional R and DeltaZ corrections to0 (estimated current exact page=existing summary); current still contributes original summary to Ohat_all/mixed denominator. Actual evaluation uses unchanged raw current KV. Report selected current/history actions; this is one conservative lifecycle diagnostic, not proof about all current error estimators.',metrics='all14760 legal swaps plus same common panel; M16 andQ64; actual original fullQ64 NMSE, positive-score no-op, I/B errors, I sign, rank/regret, missed M2 good actions; raw byte/FLOP estimates, no latency claims',invalid='nonpositive predicted mixed denominator at any legal candidate makes that method/panel invalid, no clipping/filtering actions',success_rule='exploratory screen, not deployment: primary BF16 M16 must retain>=80% of selected full-form M16 true gain in BOTH snapshots, positive in both. History-only same condition for streaming screen. Payload target<=3KiB/page, upper exploratory ceiling7KiB/page; satisfying this does not prove speed. No candidate chosen using horse to deploy.',sources={e:sha(DATA/'representatives'/f'{e}.pt') for e in EVENTS}))

def load(eid,device):
 snap=torch.load(DATA/'representatives'/f'{eid}.pt',weights_only=True,map_location='cpu');f=dict(np.load(DATA/'contributions'/eid/'summary_and_route_features.npz'));raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));an=dict(np.load(E13/'events'/eid/'anchors.npz'))
 def T(x):return x.to(device=device,dtype=torch.float64) if isinstance(x,torch.Tensor) else torch.as_tensor(x,device=device,dtype=torch.float64)
 assert np.array_equal(snap['ids'].numpy(),f['token_ids']);assert np.unique(f['token_ids']).size==410*64
 for n in ['keys','mean','detail','stats']:assert np.array_equal(snap['summary'][n][0].float().numpy(),f['summary_'+n])
 q=T(snap['q']);kg=T(snap['k']).reshape(410,4,16,128);vg=T(snap['v']).reshape_as(kg);kap=T(f['summary_keys']);basis=T(f['basis']);haar=T([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]);nu=T(f['summary_mean'])[:,None]+torch.einsum('ag,pak,dk->pgd',haar,T(f['summary_detail']),basis);ell=torch.einsum('qd,pgd->qpg',q,kap)/math.sqrt(128);logscale=(ell+math.log(16)).flatten(1).logsumexp(1);w=16*(ell-logscale[:,None,None]).exp();zh=w.sum(-1);nh=torch.einsum('qpg,pgd->qpd',w,nu);ohat=nh.sum(1);scale=1/T(raw['summary_mass']).sum(1);ze=T(raw['exact_mass'])*scale[:,None];et=T(raw['e'])*scale[:,None,None];dzt=T(raw['d'])*scale[:,None];dnt=(T(raw['e'])+T(raw['dense_output'])[:,None]*T(raw['d'])[...,None])*scale[:,None,None]
 assert np.max(abs(array(zh/scale[:,None])-raw['summary_mass']))<1e-10
 bases={k:T(v) for k,v in dict(np.load(OUT/'bases.npz')).items()};r=np.flatnonzero(raw['risk']);r=r[r!=int(raw['own'])];a=np.flatnonzero(~raw['risk']);panels={'full_legal':(np.repeat(r,len(a)),np.tile(a,len(r))),'common':(an['remove'],an['add'])}
 truth={};trueparts={}
 for panel,(r,a) in panels.items():
  g,_,parts=gain_scores(et,ze,zh,raw['risk'],r,a,np.arange(64),details=True);truth[panel]=g*64/float(raw['C']);trueparts[panel]=parts
 assert np.max(abs(truth['common']-an['gains'][0]))<1e-10
 ztrue=np.where(raw['risk'][None],raw['exact_mass'],raw['summary_mass']).sum(1);J=float(np.sum(np.square(raw['e'][:,~raw['risk']].sum(1)).sum(-1)/ztrue**2)/float(raw['C']))
 return dict(eid=eid,device=device,T=T,raw=raw,f=f,q=q,dk=kg-kap[:,:,None],vr=vg-nu[:,:,None],nu=nu,w=w,zh=zh,nh=nh,ohat=ohat,scale=scale,ze=ze,et=et,dzt=dzt,dnt=dnt,bases=bases,panels=panels,truth=truth,trueparts=trueparts,J=J)

def direct(ctx,form,uk=None,uv=None):
 q=ctx['q'];dk=ctx['dk'];vr=ctx['vr'];w=ctx['w'];nu=ctx['nu'];delta=torch.einsum('qd,pgtd->qpgt',q,dk)/math.sqrt(128) if uk is None else torch.einsum('qk,pgtk->qpgt',q@uk,dk@uk)/math.sqrt(128)
 mean=delta.mean(-1);eta=delta-mean[...,None] if form=='centered' else delta;fac=mean.exp() if form=='centered' else torch.ones_like(mean);poly=1+eta+.5*eta.square();mass=poly.mean(-1);a0=vr.mean(2);dyn=torch.einsum('qpgt,pgtd->qpgd',poly-1,vr)/16
 if uv is not None:dyn=(dyn@uv)@uv.T
 R=w[...,None]*fac[...,None]*(a0[None]+dyn);zg=w*fac*mass;zp=zg.sum(-1);npred=(nu[None]*zg[...,None]+R).sum(2);ep=npred-ctx['nh']-ctx['ohat'][:,None]*(zp-ctx['zh'])[...,None]
 return ep,zp,dict(negative_group_mass_fraction=float((zg<=0).double().mean().item()),negative_page_mass_fraction=float((zp<=0).double().mean().item()))

def build_stats(ctx,form,uk,uv,bf16):
 z=ctx['dk']@uk;y=ctx['vr']@uv;b1=z.mean(2);s=z-b1[:,:,None] if form=='centered' else z;k=uk.shape[1];i,j=torch.triu_indices(k,k,device=z.device);phi=s[...,i]*s[...,j]
 stats=dict(b1=b1,B2=phi.mean(2),a0=ctx['vr'].mean(2),A1=torch.einsum('pgtv,pgtk->pgvk',y,s)/16,A2=torch.einsum('pgtv,pgtu->pgvu',y,phi)/16)
 if bf16:stats={n:t.bfloat16() for n,t in stats.items()}
 return stats

def from_stats(ctx,form,uk,uv,stats):
 s={n:t.double() for n,t in stats.items()};t=ctx['q']@uk/math.sqrt(128);i,j=torch.triu_indices(uk.shape[1],uk.shape[1],device=t.device);quad=t[:,i]*t[:,j]*torch.where(i==j,1.,2.);b=torch.einsum('qk,pgk->qpg',t,s['b1']);b2=torch.einsum('qu,pgu->qpg',quad,s['B2']);v1=torch.einsum('qk,pgvk->qpgv',t,s['A1']);v2=torch.einsum('qu,pgvu->qpgv',quad,s['A2']);fac=b.exp() if form=='centered' else torch.ones_like(b);mass=1+.5*b2 if form=='centered' else 1+b+.5*b2;zg=ctx['w']*fac*mass;R=ctx['w'][...,None]*fac[...,None]*(s['a0'][None]+(v1+.5*v2)@uv.T);zp=zg.sum(-1);npred=(ctx['nu'][None]*zg[...,None]+R).sum(2);ep=npred-ctx['nh']-ctx['ohat'][:,None]*(zp-ctx['zh'])[...,None]
 return ep,zp,dict(negative_group_mass_fraction=float((zg<=0).double().mean().item()),negative_page_mass_fraction=float((zp<=0).double().mean().item()))

def evaluate(ctx,name,ep,zp,scope='all_pages',meta=None):
 if scope=='history_only':
  history=torch.as_tensor(ctx['f']['page_is_history'],device=ep.device);ep=torch.where(history[None,:,None],ep,torch.zeros_like(ep));zp=torch.where(history[None],zp,ctx['zh'])
 raw=ctx['raw'];dn=ep+ctx['ohat'][:,None]*(zp-ctx['zh'])[...,None];sc=ctx['scale'];en=ep/sc[:,None,None];et=ctx['et']/sc[:,None,None];dnp=dn/sc[:,None,None];dnt=ctx['dnt']/sc[:,None,None];dzp=(zp-ctx['zh'])/sc[:,None];dzt=ctx['dzt']/sc[:,None];risk=torch.as_tensor(raw['risk'],device=ep.device);page=[]
 for label,mask in [('all',torch.ones_like(risk)),('risk_unselected',~risk),('history_unselected',(~risk)&torch.as_tensor(ctx['f']['page_is_history'],device=ep.device)),('current_unselected',(~risk)&~torch.as_tensor(ctx['f']['page_is_history'],device=ep.device))]:
  def nmse(a,b):return float((a[:,mask]-b[:,mask]).square().sum().item()/(b[:,mask].square().sum().item()+1e-30))
  page.append(dict(event_id=ctx['eid'],method=name,scope=scope,page_subset=label,net_nmse=nmse(en,et),deltaN_nmse=nmse(dnp,dnt),deltaZ_nmse=nmse(dzp,dzt)))
 perf=[];selected=[];scores={}
 for panel,(r,a) in ctx['panels'].items():
  truth=ctx['truth'][panel];tr=ctx['trueparts'][panel]
  for M,qs in [(16,ctx['f']['probe_indices'].astype(int)),(64,np.arange(64))]:
   score,bad,parts=gain_scores(ep,zp,ctx['zh'],raw['risk'],r,a,qs,details=True);key=name+'__'+scope+'__'+panel+'__Q'+str(M)
   if score is None:perf.append(dict(event_id=ctx['eid'],method=name,scope=scope,panel=panel,probes=M,status='INVALID_DENOMINATOR',invalid_actions=bad));continue
   valid=abs(tr['I'])>1e-12;imse=float(np.sum((parts['I']-tr['I'])**2)/(np.sum(tr['I']**2)+1e-30));bmse=float(np.sum((parts['B']-tr['B'])**2)/(np.sum(tr['B']**2)+1e-30));met=metrics(score[None],truth[None],np.array([ctx['J']]));perf.append(dict(event_id=ctx['eid'],method=name,scope=scope,panel=panel,probes=M,status='VALID',invalid_actions=0,I_nmse=imse,B_nmse=bmse,I_sign_accuracy=float(np.mean((parts['I'][valid]>0)==(tr['I'][valid]>0))),**met));j=int(score.argmax());selected.append(dict(event_id=ctx['eid'],method=name,scope=scope,panel=panel,probes=M,remove=int(r[j]),add=int(a[j]),remove_history=bool(ctx['f']['page_is_history'][r[j]]),add_history=bool(ctx['f']['page_is_history'][a[j]]),execute=bool(score[j]>0),pred_score=float(score[j]),true_gain=float(truth[j]),nmse=ctx['J']));scores[key]=score
 return dict(page_metrics=page,performance=perf,selected=selected,scores=scores,metadata=meta or {})
