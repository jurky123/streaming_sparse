from ..e20_compressed_response.core import *
from ..e12_swap_prediction.core import objective,swap_labels
import argparse
OUT=ROOT/'artifacts/e22_active_probing_v1';SEEDS=[17,29,43,71,101]
def setup():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():return
 perms={}
 for eid in EVENTS:
  for s in SEEDS:
   rng=np.random.default_rng(np.random.SeedSequence([s,int(hashlib.sha256(eid.encode()).hexdigest()[:8],16)]));perms[eid+'__'+str(s)]=np.stack([rng.permutation(16) for _ in range(410*4)]).reshape(410,4,16)
 np.savez(OUT/'sampling_plan.npz',**perms)
 dump(OUT/'protocol.json',dict(created=time.time(),scope='Stage1 only two existing snapshots, M16, Q64 real evaluation, Page64, K41 own protected, one swap only, all410 pages sampled. No neural training, new data or production change.',seeds=SEEDS,b=[1,2,4],sampling='independent per-event/seed/page/group uniform WITHOUT replacement permutations, nested first b; no truth-based selection; plan hash locked before evaluation',plan_sha=sha(OUT/'sampling_plan.npz'),panels='PRIMARY Medium:16 lowest Risk selected non-own removals and32 highest Risk unselected additions (stable sort ties pageID); SECONDARY full14760 legal swaps to diagnose selection noise',independent='same sampled N/Z, same estimated Dense center; per-page s_p=mean_M ||e_p||^2 / Ztilde_D^2; one swap maximizes s_add-s_remove within same pool, positive score else no-op. Does NOT reroute41 pages; this controls one-action strength and uses no set interaction.',set='estimated Dense output from all sampled pages, candidate mixed outputs from same sampled N/Z and original summary; optimize squared difference averaged16 Queries, no true Dense/trueC in policy',noop='base:positive predicted gain only. Independent-repeat diagnostic pairs17->29 and43->71: proposed action from first seed accepted only if second seed scores SAME action positive; second samples are extra cost, no true-label gating. Seed101 only primary variance.',information_gate='For b1 or b2: ProbeSet mean true gain>0 in EACH snapshot, positive in>=4/5 seeds EACH, mean true gain strictly greater than ProbeIndependent EACH, primary Medium. Exploratory gate, not statistical significance.',cost_gate='No actual saved exact pages proven at fixedK41. Report extra probe byte and QK/AV costs relative to41 pages plus scoring; no expand or kernel unless actual quality-matched savings demonstrated. No stage2 if either gate fails.',stage2='Conditional only. Candidate/background weighted probing and selected-page feedback are not automatically authorized as large sweeps; retain as unrun if stage1 gates fail.',sources={e:sha(DATA/'representatives'/f'{e}.pt') for e in EVENTS}))

def run(eid,device):
 out=OUT/eid;out.mkdir(exist_ok=True);ctx=load(eid,device);raw=ctx['raw'];f=ctx['f'];qs=f['probe_indices'].astype(int);T=ctx['T'];risk=f['risk_scores'];rr=np.flatnonzero(raw['risk']);rr=rr[rr!=int(raw['own'])];aa=np.flatnonzero(~raw['risk']);rr=rr[np.argsort(risk[rr],kind='stable')[:16]];aa=aa[np.argsort(-risk[aa],kind='stable')[:32]];panels={'medium':(np.repeat(rr,len(aa)),np.tile(aa,len(rr))),'full_legal':ctx['panels']['full_legal']};truth={p:swap_labels(raw,raw['risk'],r,a) for p,(r,a) in panels.items()};trueparts={}
 for p,(r,a) in panels.items():
  _,_,trueparts[p]=gain_scores(ctx['et'],ctx['ze'],ctx['zh'],raw['risk'],r,a,qs,details=True)
 q=ctx['q'][qs];dk=ctx['dk'];v=ctx['vr']+ctx['nu'][:,:,None];w=ctx['w'][qs];nh=ctx['nh'][qs];zh=ctx['zh'][qs];perms=np.load(OUT/'sampling_plan.npz');rows=[];quality=[];scores={};audit=[]
 for seed in SEEDS:
  perm=perms[eid+'__'+str(seed)]
  for b in [1,2,4]:
   ix=torch.as_tensor(perm[:,:,:b],device=device);index=ix[...,None].expand(-1,-1,-1,128);kd=torch.gather(dk,2,index);vv=torch.gather(v,2,index);logit=torch.einsum('qd,pgtd->qpgt',q,kd)/math.sqrt(128);weights=w[...,None]*logit.exp()/b;zp=weights.sum(-1).sum(-1);npred=torch.einsum('qpgt,pgtd->qpd',weights,vv);od=npred.sum(1)/zp.sum(1)[:,None];dz=zp-zh;dn=npred-nh;e=dn-od[:,None]*dz[...,None];page_score=(e.square().sum(-1)/zp.sum(1)[:,None].square()).mean(0)
   # Persist estimates and signed residuals in a shared per-Query summary-total-Z scale.
   np.savez_compressed(out/f'probe_s{seed}_b{b}.npz',N=array(npred),Z=array(zp),deltaN=array(dn),deltaZ=array(dz),e=array(e),estimated_dense_output=array(od),page_score=array(page_score),sample_indices=perm[:,:,:b],probe_indices=qs)
   for p,(r,a) in panels.items():
    sc,bad,parts=gain_scores(e,zp,zh,raw['risk'],r,a,np.arange(16),details=True);assert bad==0 and sc is not None;ind=array(page_score)[a]-array(page_score)[r];scores[f'{seed}__{b}__{p}__set']=sc;scores[f'{seed}__{b}__{p}__independent']=ind;tr=trueparts[p];valid=abs(tr['I'])>1e-12
    quality.append(dict(event_id=eid,seed=seed,b=b,panel=p,I_sign_accuracy=float(np.mean((parts['I'][valid]>0)==(tr['I'][valid]>0))),I_nmse=float(np.sum((parts['I']-tr['I'])**2)/(np.sum(tr['I']**2)+1e-30)),B_nmse=float(np.sum((parts['B']-tr['B'])**2)/(np.sum(tr['B']**2)+1e-30))))
    for name,score in [('set',sc),('independent',ind)]:
     met=metrics(score[None],truth[p][None],np.array([ctx['J']]));j=int(score.argmax());mask=raw['risk'].copy();execute=bool(score[j]>0)
     if execute:mask[r[j]]=False;mask[a[j]]=True
     g=ctx['J']-objective(raw,mask);assert abs(g-(truth[p][j] if execute else 0))<1e-10;assert mask.sum()==41 and mask[int(raw['own'])];rows.append(dict(event_id=eid,seed=seed,b=b,panel=p,method=name,remove=int(r[j]),add=int(a[j]),execute=execute,selected_true_gain=float(truth[p][j]),**met))
    # Independent sampled mixed-output check for best and deterministic random actions.
    for j in [int(sc.argmax()),0,len(r)-1]:
     x=torch.as_tensor(raw['risk'].copy(),device=device);x[r[j]]=False;x[a[j]]=True;xp=torch.as_tensor(raw['risk'],device=device)
     def loss(x):
      ns=torch.where(x[None,:,None],npred,nh).sum(1);zs=torch.where(x[None],zp,zh).sum(1);return (od-ns/zs[:,None]).square().sum(-1).mean()
     delta=float(loss(xp)-loss(x));assert abs(delta-sc[j])<1e-10;audit.append(abs(delta-sc[j]))
 # b16 all-token limit must reproduce exact contributions and true sampled Dense output.
 ww=w[...,None]*torch.einsum('qd,pgtd->qpgt',q,dk).div(math.sqrt(128)).exp()/16;ze=ww.sum(-1).sum(-1);ne=torch.einsum('qpgt,pgtd->qpd',ww,v);od=ne.sum(1)/ze.sum(1)[:,None];er=max(float((ze-ctx['ze'][qs]).abs().max()),float((od-T(raw['dense_output'])[qs]).abs().max()));assert er<1e-10
 repeated=[]
 for s1,s2 in [(17,29),(43,71)]:
  for b in [1,2,4]:
   for p,(r,a) in panels.items():
    for name in ['set','independent']:
     first=scores[f'{s1}__{b}__{p}__{name}'];second=scores[f'{s2}__{b}__{p}__{name}'];j=int(first.argmax());ex=bool(first[j]>0 and second[j]>0);gain=float(truth[p][j]) if ex else 0;repeated.append(dict(event_id=eid,seed1=s1,seed2=s2,b=b,panel=p,method=name,execute=ex,remove=int(r[j]),add=int(a[j]),gain=gain,relative_pct=100*gain/ctx['J']))
 np.savez_compressed(out/'scores.npz',**scores);np.savez_compressed(out/'truth.npz',**{p+'_'+n:x for p,(r,a) in panels.items() for n,x in [('remove',r),('add',a),('gain',truth[p])]});dump(out/'result.json',dict(rows=rows,quality=quality,repeat_confirmation=repeated,audit=dict(max_sampled_objective_identity_error=max(audit),all_token_limit_error=er,sampling_plan_sha=sha(OUT/'sampling_plan.npz'),snapshot_sha=sha(DATA/'representatives'/f'{eid}.pt'))));print(eid,'done',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--event');p.add_argument('--device',default='cuda:0');a=p.parse_args();setup() if a.setup else run(a.event,a.device)
