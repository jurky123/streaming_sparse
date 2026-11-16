from .core import *

def run(seed):
 torch.set_num_threads(2);data=Data(seed);dest=D/f's{seed}';spec=json.loads((dest/'protocol.json').read_text());assert json.loads((dest/'layer_complete.json').read_text())['steps']=={'1':768,'15':768,'29':768};arr=np.load(dest/'small_targets.npz');chosen=json.loads((E43/f's{seed}/protocol.json').read_text())['events'];checks=[];model=model_from(dest/'aligned_final.pt')
 # Original canonical contributions, direct positive sums at fixed actions.
 for ei in [0,15,31,47]:
  d=data.load(chosen[ei]);ev=torch.load(OUT/chosen[ei],map_location='cpu',weights_only=True);key='l%sh%s'%(ev['layer'],ev['head']);q=ev['q'].cuda().float();c=contributions(q,d['s'],data.bundle.layers[key].basis,ev['log_scale'].cuda().double());qs=data.qs;small={k:v[qs] if torch.is_tensor(v) and v.shape[0]==64 else v for k,v in c.items()};N=ev['N'].cuda().double()[qs];Z=ev['Z'].cuda().double()[qs];od=ev['dense'].cuda().double()[qs];gm=m16_labels(data,d);assert np.array_equal(gm.cpu().numpy(),arr['g_M16'][2*ei]);errmax=0.
  for j in [0,127,511]:
   mask=d['mask'].clone();mask[d['r'][j]]=False;mask[d['a'][j]]=True
   from ..e24_joint_representatives.core import mixed
   before=mixed(small,N,Z,d['mask']);after=mixed(small,N,Z,mask);direct=((before-od).square().sum()-(after-od).square().sum())/(16*d['norm']);err=float(abs(direct-gm[j]));assert torch.allclose(direct,gm[j],rtol=1e-5,atol=1e-9);errmax=max(errmax,err)
  checks.append(dict(event=chosen[ei],M16_target_direct_error=errmax))
 # Full data event identity and no unintended teacher inputs: inference uses only summary and F.
 for phase in ['frozen','trained']:
  assert json.loads((dest/f'{phase}_evaluation_complete.json').read_text())['events']==1440
  with (dest/f'{phase}_decisions.csv').open() as f:rr=list(csv.DictReader(f))
  assert len(rr)==5760
 # Directly verify the worst executed actions for every new model/split.
 from ..e24_joint_representatives.core import mixed
 for phase in ['frozen','trained']:
  with (dest/f'{phase}_decisions.csv').open() as f:rr=list(csv.DictReader(f))
  groups={}
  for r in rr:
   if r['policy']=='Set':groups.setdefault((r['arm'],r['split']),[]).append(r)
  for (arm,split),v in groups.items():
   worst=min(v,key=lambda r:float(r['executed_gain']));d=data.load(worst['event']);ev=torch.load(OUT/worst['event'],map_location='cpu',weights_only=True);key='l%sh%s'%(ev['layer'],ev['head']);c=contributions(ev['q'].cuda().float(),d['s'],data.bundle.layers[key].basis,ev['log_scale'].cuda().double());N,Z,od=[ev[k].cuda().double() for k in ['N','Z','dense']];mask=d['mask'].clone()
   if worst['execute']=='True':mask[int(worst['remove'])]=False;mask[int(worst['add'])]=True
   direct=float(objective(c,N,Z,od,d['mask'])-objective(c,N,Z,od,mask));legacy=float(worst['executed_gain']);assert np.isclose(direct,legacy,rtol=1e-5,atol=1e-9),(arm,split,direct,legacy)
   checks.append(dict(event=worst['event'],arm=arm,split=split,direct_gain=direct,legacy_gain=legacy,gain_error=abs(direct-legacy)))
 data.check_frozen();table(dest/'audit_checks.csv',checks);dump(dest/'audit.json',dict(passed=True,seed=seed,alignment_direct_checks=len(checks),full_phase_seed_events=2880,active_layer_steps=768,optimizer_updates_total=2304,Test=False,raw_attention=False));print('AUDIT PASS',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
