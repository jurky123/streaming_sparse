from .core import *

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run(seed):
 torch.set_num_threads(2);data,states,groups,initial=build(seed,verify=True);dest=D/f's{seed}';assert (dest/'complete.json').exists();rows=[];npz=np.load(dest/'scores_2304.npz');er=read(dest/'evaluation_2304.csv')
 for name in ARMS:
  model=Lookup(groups,initial).cuda() if name.startswith('L') else model_from(E29/f's{seed}/selected.pt');model.load_state_dict(torch.load(dest/f'{name}_final.pt',map_location='cuda',weights_only=True))
  rr=[r for r in er if r['arm']==name and r['policy']=='Set'];worst=sorted(rr,key=lambda x:float(x['executed_gain']))[:2];selected=sorted({0,31,63,93,95}|{int(x['state']) for x in worst})
  for i in selected:
   d=states[i];rp,bp=outputs(model,d);sc,_=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);old=torch.tensor(npz[name][i],device='cuda');assert torch.equal(sc,old)
   alt,_=safe_predictions(reconstruct(rp,bp,d['lm'],d['center']),bp,d['lm'],d['mask'],d['r'],d['a'],d['norm']);err=float((sc-alt).abs().max());assert torch.allclose(sc,alt,rtol=1e-6,atol=1e-8)
   ev=torch.load(OUT/d['meta']['event'],map_location='cpu',weights_only=True);key='l%sh%s'%(ev['layer'],ev['head']);c=contributions(ev['q'].cuda().float(),d['s'],data.bundle.layers[key].basis,ev['log_scale'].cuda().double());j=int(sc.argmax());mask=d['mask'].clone();ex=bool(sc[j]>0)
   if ex:mask[d['r'][j]]=False;mask[d['a'][j]]=True
   assert int(mask.sum())==int(d['mask'].sum()) and bool(mask[d['own']]);N,Z,od=[ev[k].cuda().double() for k in ['N','Z','dense']];direct=float(objective(c,N,Z,od,d['mask'])-objective(c,N,Z,od,mask));legacy=float(d['g'][j]) if ex else 0.;assert np.isclose(direct,legacy,rtol=1e-5,atol=1e-9)
   rows.append(dict(seed=seed,arm=name,state=i,score_error=err,gain_error=abs(direct-legacy),direct_gain=direct,finite=True,count_own=True))
 # Old E43 prediction and actual action/gain replay.
 old=read(E43/f's{seed}/evaluation_1.csv');new=read(dest/'evaluation_-1.csv');checks=0
 for r in new:
  arm={'E43_P':'P','E43_L':'L'}[r['arm']];o=next(z for z in old if z['arm']==arm and z['state']==r['state'] and z['policy']==r['policy']);assert abs(float(o['executed_gain'])-float(r['executed_gain']))<1e-9;checks+=1
 data.check_frozen();table(dest/'independent_checks.csv',rows);dump(dest/'audit.json',dict(passed=True,independent_checks=len(rows),E43_replay_checks=checks,identity_audit=json.loads((dest/'identity_audit.json').read_text()),sparse_clock_verified=True,Test=False,max_score_error=max(r['score_error'] for r in rows),max_gain_error=max(r['gain_error'] for r in rows)));print('AUDIT PASS',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
