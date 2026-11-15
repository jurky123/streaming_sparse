"""Frozen E24 diagnostics. Never loads the test split."""
from ..e24_joint_representatives.train import *
from scipy.stats import pearsonr
import itertools
E25=ROOT/'artifacts/e25_decision_diagnostics_v1'

def checkpoint(group,seed):
 if group=="D":return E25/f"D_s{seed}"/"selected.pt"
 return OUT/'training'/f'{"A_matched" if group=="A" else "C"}_s{seed}'/'selected.pt'

def bundle_for(group,seed):
 b=Bundle();ck=torch.load(checkpoint(group,seed),weights_only=False,map_location='cuda');b.layers.load_state_dict(ck['layers']);b.reader.load_state_dict(ck['reader']);b.eval();return b

def project_gains(c,d,mask,r,a,qs,B=None):
 # Algebraically the online-centred t/u objective; evaluate in output space
 # to avoid subtracting two enormous centred corrections.
 N,Z,od=d['N'][qs],d['Z'][qs],d['dense'][qs]
 nh,zh=c['nh'][qs],c['zh'][qs]
 ns=torch.where(mask[None,:,None],N,nh).sum(1);zs=torch.where(mask[None],Z,zh).sum(1)
 base=od-ns/zs[:,None]
 if B is not None:base=base@B.double()
 base=base.square().sum();den=od.square().sum()+1e-12
 ans=[]
 for start in range(0,len(r),64):
  rr,aa=r[start:start+64],a[start:start+64]
  n=ns[:,None]-(N-nh)[:,rr]+(N-nh)[:,aa];z=zs[:,None]-(Z-zh)[:,rr]+(Z-zh)[:,aa]
  assert torch.isfinite(z).all() and (z>0).all()
  e=od[:,None]-n/z[...,None]
  if B is not None:e=e@B.double()
  ans.append((base-e.square().sum((0,2)))/den)
 return torch.cat(ans)

def corr(x,y):
 x=np.asarray(x).ravel();y=np.asarray(y).ravel()
 if len(x)<2 or np.std(x)==0 or np.std(y)==0:return None
 return float(np.corrcoef(x,y)[0,1])

def conditional_metrics(t,p):
 t=np.asarray(t);p=np.asarray(p);tc=t-t.mean(0);pc=p-p.mean(0)
 out=dict(conditional_corr=corr(tc,pc),ranking_interaction_corr=corr(tc-tc.mean(1,keepdims=True),pc-pc.mean(1,keepdims=True)),overall_corr=corr(t,p))
 difft=[];diffp=[];tsigns=[];psigns=[];rankt=[];rankp=[]
 n=t.shape[1];rng=np.random.default_rng(1701);ia=rng.integers(n,size=min(1024,n*2));ib=rng.integers(n,size=len(ia));keep=ia!=ib;ia=ia[keep];ib=ib[keep]
 for i,j in itertools.combinations(range(len(t)),2):
  difft.append(t[j]-t[i]);diffp.append(p[j]-p[i]);valid=(abs(t[i])>1e-12)&(abs(t[j])>1e-12)
  tsigns.extend(((t[i][valid]>0)!=(t[j][valid]>0)).tolist());psigns.extend(((p[i][valid]>0)!=(p[j][valid]>0)).tolist())
  ti=t[i,ia]-t[i,ib];tj=t[j,ia]-t[j,ib];pi=p[i,ia]-p[i,ib];pj=p[j,ia]-p[j,ib];v=(abs(ti)>1e-12)&(abs(tj)>1e-12)
  rankt.extend(((ti[v]>0)!=(tj[v]>0)).tolist());rankp.extend(((pi[v]>0)!=(pj[v]>0)).tolist())
 out['delta_corr']=corr(np.concatenate(difft),np.concatenate(diffp))
 for name,x,y in [('sign_flip',tsigns,psigns),('rank_flip',rankt,rankp)]:
  x=np.array(x,dtype=bool);y=np.array(y,dtype=bool);out.update({name+'_n':len(x),name+'_true':int(x.sum()),name+'_pred':int(y.sum()),name+'_tp':int((x&y).sum()),name+'_correct':int((x==y).sum())})
 return out

@torch.no_grad()
def run(group,seed):
 torch.set_num_threads(2);dest=E25/f'{group}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 b=bundle_for(group,seed);B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long();allq=torch.arange(64,device='cuda');rows=[];csrows=[];curves=[];start=time.time();identitymax=0.;p0max=0.
 oldrows={x['event']:x for x in json.loads((checkpoint(group,seed).parent/f'dev_epoch{torch.load(checkpoint(group,seed),weights_only=False,map_location="cpu")["epoch"]}.json').read_text())}
 for idx,path in enumerate(files('development')):
  d=load(path);m=b.layers[d['key']];s=b.make(d,group);c=contributions(d['q'],s,m.basis,d['log_scale']);mask,r,a=route(d['q'][qs],s,m,d['own']);cp={n:x[qs] for n,x in c.items()};z,u,beta=b.reader(d['q'][qs],s,cp);pg=predicted(z,u,cp['mass'],mask,r,a,beta)/(d['dense'][qs].square().sum()/16+1e-12)
  J=float(objective(c,d['N'],d['Z'],d['dense'],mask));g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);g3=project_gains(c,d,mask,r,a,allq);identitymax=max(identitymax,float((g-g3).abs().max()));assert torch.allclose(g,g3,atol=1e-9,rtol=1e-6)
  scores={'P0':pg,'P1':project_gains(c,d,mask,r,a,qs,B),'P2':project_gains(c,d,mask,r,a,qs),'P3':g3};event=str(path.relative_to(OUT));oracle=max(0.,float(g.max()));eventrows=[]
  for name,score in scores.items():
   assert torch.isfinite(score).all();j=int(score.argmax());ex=bool(score[j]>0);true=float(g[j]);row=dict(event=event,trajectory=d['ev']['trajectory'],group=group,seed=seed,method=name,initial=J,oracle_gain=oracle,proposal_gain=true,executed_gain=true if ex else 0.,hindsight_gain=max(0.,true),execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j]),layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head']);rows.append(row);eventrows.append(row)
  ref=oldrows[event];p0max=max(p0max,abs(eventrows[0]['executed_gain']-ref['set_gain']));assert eventrows[0]['execute']==ref['execute'] and eventrows[0]['remove']==ref['set_remove'] and eventrows[0]['add']==ref['set_add'];assert int(mask.sum())==max(1,int(len(mask)*.1)) and bool(mask[d['own']])
  # Reconstruct E24-style teacher/random counterfactual states under this frozen summary.
  f,_=RouteOps.features(d['q'][qs][None],s);risk=(f*m.theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy();states=[mask];names=['risk'];rng=np.random.default_rng(17000+idx)
  for mode in [6,7,8,9]:
   st,_,_=getstate(c,d,mask.clone(),r,a,mode,risk,rng);states.append(st);names.append({6:'oracle1',7:'oracle2',8:'random1',9:'random2'}[mode])
  # Keep the original Medium pairs that are legal in every counterfactual state.
  keep=torch.ones_like(r,dtype=torch.bool)
  for st in states:keep &= st[r]&~st[a]
  rr,aa=r[keep],a[keep];assert len(rr)>0
  ts=[];ps=[]
  for st in states:
   assert int(st.sum())==int(mask.sum()) and st[d['own']]
   ts.append(gains(c,d['N'],d['Z'],d['dense'],st,rr,aa));ps.append(predicted(z,u,cp['mass'],st,rr,aa,beta)/(d['dense'][qs].square().sum()/16+1e-12))
  ts=torch.stack(ts).cpu().numpy();ps=torch.stack(ps).cpu().numpy();assert np.isfinite(ts).all() and np.isfinite(ps).all()
  cr=dict(event=event,trajectory=d['ev']['trajectory'],group=group,seed=seed,initial=J,common_actions=len(rr),unique_states=len({tuple(torch.where(x)[0].tolist()) for x in states}),**conditional_metrics(ts,ps));csrows.append(cr)
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=np.array(event),states=torch.stack(states).cpu().numpy(),state_names=np.array(names),remove=rr.cpu().numpy(),add=aa.cpu().numpy(),true_gain=ts,predicted_gain=ps,initial_remove=r.cpu().numpy(),initial_add=a.cpu().numpy(),**{k:v.cpu().numpy() for k,v in scores.items()},true_initial_gain=g.cpu().numpy())
  if idx%24==0:dump(dest/'status.json',dict(done=idx,total=288,seconds=time.time()-start));print(group,seed,idx,flush=True)
 dump(dest/'ladder.json',rows);dump(dest/'conditional.json',csrows);dump(dest/'complete.json',dict(group=group,seed=seed,events=288,seconds=time.time()-start,checkpoint_sha=sha(checkpoint(group,seed)),fit_sha=sha(OUT/'fit.pt'),manifest_sha=sha(OUT/'dataset_manifest.json'),p3_direct_gain_max_error=identitymax,p0_replay_max_error=p0max,test_accessed=False));print('DONE',group,seed,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['A','C','D'],required=True);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();run(args.group,args.seed)
