"""Post-run provenance, online invariants, and direct positive-mixture checks."""
from .core import *
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
@torch.no_grad()
def run(seed):
 torch.set_num_threads(2);data=Data(seed);rows=[];T1=D/f'T1_s{seed}';assert (T1/'evaluation_complete.json').exists();ck=torch.load(T1/'epoch1.pt',map_location='cpu',weights_only=False);assert ck['updates']==2304 and ck['epoch']==1
 group=ck['optimizer']['param_groups'][0];assert group['lr']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0;assert all(float(s['step'])==2304 for s in ck['optimizer']['state'].values())
 initial=torch.load(T1/'initial.pt',map_location='cpu',weights_only=True);original=torch.load(E29/f's{seed}/selected.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,original[k]) for k,v in initial.items())
 contract_rec=json.loads((D/f'contract_s{seed}.json').read_text());assert sha(E39/f'T0_s{seed}/epoch1.pt')==contract_rec['T0_sha'];assert sha(E39/f'schedule_s{seed}.json')==contract_rec['schedule_sha']
 for arm in ['T0','T1']:
  dest=D/f'{arm}_s{seed}';model=model_from(E39/f'T0_s{seed}/epoch1.pt' if arm=='T0' else dest/'epoch1.pt').eval();ds=read(dest/'decisions.csv');assert len(ds)==2880
  for split in ['train','development']:
   metas=json.loads((CACHE/f'{split}_s{seed}/metadata.json').read_text());events={m['event']:m['index'] for m in metas};rr=[r for r in ds if r['split']==split and r['policy']=='Set'];selected={metas[i]['event'] for i in [0,len(metas)//3,2*len(metas)//3]};selected.update(r['event'] for r in sorted(rr,key=lambda x:float(x['executed_gain']))[:2])
   for event in sorted(selected):
    d=data.load(event);rp,bp=model(d['s'],d['F']);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);alt,_=safe_predictions(reconstruct(rp,bp,d['lm'],d['center']),bp,d['lm'],d['mask'],d['r'],d['a'],d['norm']);score_error=float((sc-alt).abs().max());assert torch.allclose(sc,alt,atol=1e-9,rtol=1e-7)
    pp=np.load(dest/('%s_panel_%04d.npz'%(split,events[event])));assert np.array_equal(pp['scores'],sc.cpu().numpy());j=int(sc.argmax());executed=bool(sc[j]>0);newmask=d['mask'].clone()
    if executed:newmask[d['r'][j]]=False;newmask[d['a'][j]]=True
    assert int(newmask.sum())==int(d['mask'].sum()) and bool(newmask[d['own']])
    # Direct sum over positive page contributions, avoiding subtractive update.
    ev=torch.load(OUT/event,map_location='cpu',weights_only=True);key='l%sh%s'%(ev['layer'],ev['head']);c=contributions(ev['q'].cuda().float(),d['s'],data.bundle.layers[key].basis,ev['log_scale'].cuda().double());N,Z,od=[ev[n].cuda().double() for n in ['N','Z','dense']];direct=float(objective(c,N,Z,od,d['mask'])-objective(c,N,Z,od,newmask));legacy=float(d['g'][j]) if executed else 0.
    assert np.isclose(direct,legacy,atol=1e-9,rtol=1e-5),(event,direct,legacy)
    rows.append(dict(seed=seed,arm=arm,split=split,event=event,score_error=score_error,direct_gain=direct,incremental_gain=legacy,gain_error=abs(direct-legacy),count_own=True))
 data.check_frozen();table(D/f'audit_events_s{seed}.csv',rows);dump(D/f'audit_s{seed}.json',dict(passed=True,seed=seed,events=len(rows),optimizer_steps=2304,optimizer_reset_and_settings_verified=True,T0_unchanged=True,initial_matches_E29=True,final_epoch=1,weight_gate=True,raw_KV=False,test=False,max_score_error=max(r['score_error'] for r in rows),max_gain_error=max(r['gain_error'] for r in rows),gain_tolerance=dict(rtol=1e-5,atol=1e-9)));print('AUDIT PASS',seed,len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
