"""E46: fixed target alignment and shared-layer bridge, Test sealed."""
from ..e44_fit_diagnostics import core as fit
from ..e42_local_coverage.core import *
D=ROOT/'artifacts/e46_supervision_bridge_v1';E43=ROOT/'artifacts/e43_fit_v1';E44=ROOT/'artifacts/e44_fit_diagnostics_v1'
def protocol(seed):
 dest=D/f's{seed}';dest.mkdir(parents=True,exist_ok=True);dump(dest/'protocol.json',dict(seed=seed,A='E43 exact48/96; 2304 updates, original joint losses; only set target M16 instead of Q64; original Q64 gain normalization held fixed; compare original E43 P',A_units='M16 true output squared-error difference averaged across16 / original Q64 mean dense energy; same as predicted score units',B='Frozen E43 P and E44 P_set on all train/development Risk events; stratify fitted48, unused chunk5-step0 same head, other heads/steps/chunks, dev',C='3 layer-specific E29 initialized Readers, 2304 total E39-T0 scheduled updates; eachlayer768, independent Adam; compare reused E39-T0 (shared) same budget',C_boundary='3x stored Reader weights, same active FLOPs; separates layer distributions but changes parameter sharing and optimizer clocks; does not establish coordinate mismatch cause',optimizer='Adam reset lr1e-4 betas .9/.999 eps1e-8 wd0 clip1; frozen C/phi/B',checkpoint='final only; no dev selection',test=False,newKV=False,query_selection='not executed: full Q64 summary contrast is not already free online; cost/observable score must be defined before testing'))

class LayerReader(torch.nn.Module):
 def __init__(self,seed):
  super().__init__();self.layers=torch.nn.ModuleDict({str(l):model_from(E29/f's{seed}/selected.pt') for l in [1,15,29]})
 def active(self,d):return self.layers[str(d['meta']['layer'])]

@torch.no_grad()
def m16_labels(data,d):
 ev=torch.load(OUT/d['meta']['event'],map_location='cpu',weights_only=True);q=ev['q'].cuda().float();key='l%sh%s'%(ev['layer'],ev['head']);c=contributions(q,d['s'],data.bundle.layers[key].basis,ev['log_scale'].cuda().double());qs=data.qs;small={k:(v[qs] if torch.is_tensor(v) and v.shape[0]==64 else v) for k,v in c.items()};od=ev['dense'].cuda().double()[qs];g=gains(small,ev['N'].cuda().double()[qs],ev['Z'].cuda().double()[qs],od,d['mask'],d['r'],d['a']);return g*(od.square().sum()+1e-12)/(16*d['norm'])

def train_small(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);protocol(seed);data,states,groups,init=fit.build(seed,verify=False);dest=D/f's{seed}';model=model_from(E29/f's{seed}/selected.pt');opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8);curves=[];labels=[]
 for d in states:d['gm']=m16_labels(data,d);labels.append(d['gm'].cpu().numpy())
 np.savez_compressed(dest/'small_targets.npz',g_M16=np.stack(labels),g_Q64=np.stack([d['g'].cpu().numpy() for d in states]))
 def evaluate(step):
  rows=[];scores=[]
  with torch.no_grad():
   for i,d in enumerate(states):
    rp,bp=model(d['s'],d['F']);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);scores.append(sc.cpu().numpy())
    for policy,score in [('Set',sc),('Independent',ind)]:
     j=int(score.argmax());ex=bool(score[j]>0);rows.append(dict(seed=seed,step=step,state=i,event=d['meta']['event'],layer=d['meta']['layer'],policy=policy,initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=float(d['g'][j]) if ex else 0.,execute=ex,proposal_gain=float(d['g'][j]),hindsight_gain=max(0.,float(d['g'][j])),local_mse=float((rp-d['rtrue']).square().mean()),local_mass_mse=float((d['mass'][...,None]*(rp-d['rtrue'])).square().mean()),beta_mae=float((bp-d['btrue']).abs().mean())))
  table(dest/f'aligned_eval_{step}.csv',rows);np.savez_compressed(dest/f'aligned_scores_{step}.npz',scores=np.stack(scores))
 evaluate(0)
 for u in range(2304):
  d=states[u%96];rp,bp=model(d['s'],d['F']);ls,bs,gs,pg,_=losses(rp,bp,d['rtrue'],d['btrue'],d['mass'],d['lm'],d['center'],d['mask'],d['r'],d['a'],d['g'],d['norm'],data.scale,data.fit);sl=original_set_loss(pg,d['gm'],gs);loss=ls[0]+ls[1]+sl;opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();curves.append(dict(update=u,local=float(ls[0]),quality=float(ls[1]),setloss=float(sl),gradnorm=float(gn)))
  if (u+1)%768==0:evaluate(u+1);print('ALIGN',seed,u+1,flush=True)
 torch.save(model.state_dict(),dest/'aligned_final.pt');table(dest/'aligned_curves.csv',curves);data.check_frozen();dump(dest/'aligned_complete.json',dict(seed=seed,updates=2304,test=False,normalization_held_fixed=True));print('ALIGN DONE',seed,flush=True)

def train_layer(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);protocol(seed);data=Data(seed);model=LayerReader(seed);opts={k:torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8) for k,m in model.layers.items()};schedule=json.loads((E39/f'schedule_s{seed}.json').read_text());assert len(schedule)==2304;dest=D/f's{seed}';curves=[]
 for st in schedule:
  d=data.load(st['event'],st);m=model.active(d);ls,_,_,_=loss_terms(m,d,data,False);loss=sum(ls);opt=opts[str(d['meta']['layer'])];opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();curves.append(dict(update=st['update'],layer=d['meta']['layer'],local=float(ls[0]),quality=float(ls[1]),setloss=float(ls[2]),gradnorm=float(gn)))
  if (st['update']+1)%768==0:print('LAYER',seed,st['update']+1,flush=True)
 torch.save(model.state_dict(),dest/'layer_final.pt');table(dest/'layer_curves.csv',curves);steps={k:int(next(iter(o.state.values()))['step']) for k,o in opts.items()};assert set(steps.values())=={768};data.check_frozen();dump(dest/'layer_complete.json',dict(seed=seed,total_updates=2304,steps=steps,parameters=sum(p.numel() for p in model.parameters()),active_parameters=51157,test=False,schedule_sha=sha(E39/f'schedule_s{seed}.json')));print('LAYER DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['align','layer'],required=True);a=p.parse_args();{'align':train_small,'layer':train_layer}[a.mode](a.seed)
