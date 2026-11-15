"""Candidate-only Key quality probes, fixed local response, no training."""
from ..e30_beta_diagnostics.run import *
from ..e25_decision_diagnostics.report import aggregate
import hashlib,collections
E30=DEST
DEST=ROOT/'artifacts/e32_candidate_key_probe_v1'
SAMPLE_SEEDS=[17,29,43,71,101];BS=[1,2,4]
BASES=['E29','Candidate-True','Full-True']

def permutations(ids,trajectory,layer,head,sampling_seed):
 ans=[]
 for page in ids:
  h=hashlib.sha256(f'{trajectory}|{layer}|{head}|{sampling_seed}'.encode()+page.contiguous().numpy().tobytes()).digest()
  rng=np.random.default_rng(int.from_bytes(h[:8],'little'));ans.append(np.stack([rng.permutation(16) for _ in range(4)]))
 return np.stack(ans).astype(np.uint8)

def plan():
 dest=DEST/'plans';dest.mkdir(parents=True,exist_ok=True);cache={};items=[]
 if (dest/'manifest.json').exists():print('PLAN ALREADY FROZEN');return
 for i,path in enumerate(files('development')):
  ev=torch.load(path,map_location='cpu',weights_only=True);kp=path.parent/ev['kv_file']
  if kp not in cache:cache[kp]=torch.load(kp,map_location='cpu',weights_only=True)['ids']
  ids=cache[kp];arrays={f's{s}':permutations(ids,ev['trajectory'],ev['layer'],ev['head'],s) for s in SAMPLE_SEEDS};np.savez_compressed(dest/f'{i:04d}.npz',event=str(path.relative_to(OUT)),ids_sha=digest(ids),**arrays);items.append(dict(index=i,event=str(path.relative_to(OUT)),pages=len(ids),ids_sha=digest(ids),plan_sha=sha(dest/f'{i:04d}.npz')))
 dump(dest/'manifest.json',dict(events=items,numpy_version=np.__version__,sampling_seeds=SAMPLE_SEEDS,protocol_sha=sha(DEST/'protocol.json'),selection_uses_only_metadata_and_ids=True));print('PLAN LOCKED',len(items),flush=True)

def probe_beta(q,sampled_keys,summary_logz,b):
 # Deliberately accepts only probes, sampled K and summary logZ; no V/teacher.
 assert sampled_keys.shape[1:3]==(4,b)
 logits=torch.einsum('qd,pgtd->qpgt',q.double(),sampled_keys.double())/math.sqrt(128)
 estimate=logits.flatten(-2).logsumexp(-1)+math.log(16/b)
 beta=estimate-summary_logz
 assert torch.isfinite(beta).all()
 return beta,estimate

@torch.no_grad()
def run(seed,limit=None):
 torch.set_num_threads(2);dest=DEST/f's{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'complete.json').exists()
 manifest=json.loads((DEST/'plans'/'manifest.json').read_text());assert manifest['protocol_sha']==sha(DEST/'protocol.json')
 bundle=bundle_for('C',seed);qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[];quality=[];conditions=[];costs=[];audit=[];start=time.time();paths=files('development')
 for idx,path in enumerate(paths):
  if limit is not None and idx>=limit:break
  d,s,c,_,_,_,_=prepare_local(bundle,path);ev=d['ev'];x=torch.load(E30/f's{seed}'/f'beta_{idx:04d}.pt',map_location='cuda',weights_only=True);p=np.load(E30/f's{seed}'/f'panel_{idx:04d}.npz');sp=np.load(DEST/'plans'/f'{idx:04d}.npz');assert str(sp['event'])==x['event'];assert manifest['events'][idx]['plan_sha']==sha(DEST/'plans'/f'{idx:04d}.npz')
  mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');cand=torch.unique(torch.cat([r,a]),sorted=True);assert len(cand)==48 and d['own'] not in cand
  lm=x['logmass'][qs];rp=x['predicted_response'][qs];center=x['center'][qs];bp=x['betas']['Predicted'][qs];bt=x['beta_true'][qs];summary_logz=(c['logit'][qs]+math.log(16)).logsumexp(-1);norm=d['dense'][qs].square().sum()/16+1e-12
  logmass_check=summary_logz-summary_logz.logsumexp(-1)[:,None];assert torch.allclose(logmass_check,lm,atol=1e-8,rtol=1e-8)
  g=torch.as_tensor(p['true_initial_gain'],device='cuda');J=float(objective(c,d['N'],d['Z'],d['dense'],mask));candidate_mask=torch.zeros_like(mask);candidate_mask[cand]=True
  bb=bp.clone().double();bb[:,cand]=bt[:,cand];configs=[('E29',None,None,bp),('Candidate-True',None,None,bb),('Full-True',None,None,bt)];sample_cache={};kg=d['k'][0].reshape(-1,4,16,128)
  for sampling_seed in SAMPLE_SEEDS:
   perm=torch.as_tensor(sp[f's{sampling_seed}'][cand.cpu().numpy()].astype(np.int64),device='cuda')
   for b in BS:
    offsets=perm[:,:,:b];sampled=kg[cand].gather(2,offsets[...,None].expand(-1,-1,-1,128));est,lz=probe_beta(d['q'][qs],sampled,summary_logz[:,cand],b);beta=bp.clone().double();beta[:,cand]=est;method=f'KeyProbe-b{b}';key=f'{method}-r{sampling_seed}';configs.append((method,sampling_seed,b,beta));sample_cache[key]=est.cpu().numpy()
  out={};init={};met=dict(seed=seed,event=x['event'],**{k:ev[k] for k in ['trajectory','layer','head','chunk','step','tile']})
  rr=torch.as_tensor(p['remove'],device='cuda');aa=torch.as_tensor(p['add'],device='cuda')
  for method,sampling_seed,b,beta in configs:
   key=method if sampling_seed is None else f'{method}-r{sampling_seed}'
   # Positive constant norm only expresses offline NMSE units; verify raw policy independently.
   sc,ind=local_scores(rp,beta,lm,center,mask,r,a,norm);init[key]=sc.cpu().numpy()
   for policy,score in [('Set',sc),('Independent',ind)]:
    j=int(score.argmax());execute=bool(score[j]>0);new=mask.clone()
    if execute:new[r[j]]=False;new[a[j]]=True
    assert int(new.sum())==int(mask.sum()) and bool(new[d['own']])
    rows.append(dict(**met,method=method,sampling_seed=sampling_seed,b=b,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),executed_gain=float(g[j]) if execute else 0.,execute=execute,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j]),regret=max(0.,float(g.max()))-(float(g[j]) if execute else 0.)))
   error=beta[:,cand].double()-bt[:,cand];m=weighted_metrics(beta,bt,lm,candidate_mask);qrow=dict(**met,method=method,sampling_seed=sampling_seed,b=b,beta_signed_bias=float(error.mean()),**m,gain_pearson=stablecorr(sc.cpu().numpy(),g.cpu().numpy()),gain_spearman=float(spearmanr(sc.cpu().numpy(),g.cpu().numpy()).statistic),positive_recall=float(((sc>0)&(g>1e-12)).sum()/(g>1e-12).sum().clamp_min(1)),sign_accuracy=float(((sc>0)==(g>1e-12)).double().mean()))
   quality.append(qrow)
   ps=np.stack([local_scores(rp,beta,lm,center,torch.as_tensor(st,device='cuda'),rr,aa,norm)[0].cpu().numpy() for st in p['states']]);out[key]=ps;conditions.append(dict(**met,method=method,sampling_seed=sampling_seed,b=b,**conditional_metrics(p['true_gain'],ps)))
  # Fixed E30 controls, stable same code path; explicit background invariant.
  for key,oldkey in [('E29','Predicted'),('Candidate-True','Candidate-True'),('Full-True','True')]:assert np.allclose(init[key],p['initial_'+oldkey],rtol=1e-8,atol=1e-10),(seed,idx,key)
  for method,rs,b,beta in configs:
   if b is not None:assert torch.equal(beta[:,~candidate_mask],bp[:,~candidate_mask].double())
  costs.append(dict(**met,pages=len(mask),exact_pages=int(mask.sum()),candidate_pages=len(cand),probes=len(qs),dimension=128,bytes_per_scalar=2))
  audit.append(dict(**met,logmass_absolute_error=float((logmass_check-lm).abs().max()),candidate_pages=48,no_own_probe=True,background_unchanged=True,plan_sha=manifest['events'][idx]['plan_sha']))
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=x['event'],candidate_pages=cand.cpu().numpy(),summary_logz_candidate=summary_logz[:,cand].cpu().numpy(),true_beta_candidate=bt[:,cand].cpu().numpy(),true_gain=p['true_gain'],true_initial_gain=g.cpu().numpy(),states=p['states'],remove=p['remove'],add=p['add'],initial_remove=r.cpu().numpy(),initial_add=a.cpu().numpy(),**out,**{'initial_'+k:v for k,v in init.items()},**{'beta_'+k:v for k,v in sample_cache.items()})
  if (idx+1)%24==0:
   for name,data in [('decisions',rows),('quality',quality),('conditional',conditions),('cost_inputs',costs),('audit',audit)]:
    for row in data:
     for k,v in row.items():
      if isinstance(v,float) and not math.isfinite(v):row[k]=None
    dump(dest/(name+'.json'),data)
   dump(dest/'status.json',dict(seed=seed,events=idx+1,seconds=time.time()-start));print('EVENTS',seed,idx+1,round(time.time()-start),flush=True)
 for name,data in [('decisions',rows),('quality',quality),('conditional',conditions),('cost_inputs',costs),('audit',audit)]:dump(dest/(name+'.json'),data)
 if limit is None:
  assert len(costs)==288;dump(dest/'complete.json',dict(seed=seed,events=288,seconds=time.time()-start,training=False,test_accessed=False,reader_sha=sha(E29/f's{seed}'/'selected.pt'),C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}'/'selected.pt'),plan_manifest_sha=sha(DEST/'plans'/'manifest.json'),protocol_sha=sha(DEST/'protocol.json')))
 print('DONE',seed,len(costs),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--limit',type=int);a=p.parse_args()
 if a.plan:plan()
 else:assert a.seed in [17,29];run(a.seed,a.limit)
