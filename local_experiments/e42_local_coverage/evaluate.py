"""Full fixed-budget train/dev evaluation of final E42 arms."""
from .core import *
from ..e25_decision_diagnostics.report import aggregate

@torch.no_grad()
def evaluate(seed,arm):
 torch.set_num_threads(2);dest=D/f'{arm}_s{seed}';dest.mkdir(exist_ok=True);assert not (dest/'evaluation_complete.json').exists();data=Data(seed)
 path=E39/f'T0_s{seed}/epoch1.pt' if arm=='T0' else dest/'epoch1.pt'
 if arm=='T1':assert (dest/'training_complete.json').exists()
 model=model_from(path).eval();rows=[];info=[];replay=[];start=time.time();old_by={}
 if arm=='T0':
  for r in json.loads((E39/f'T0_s{seed}/development_epoch1.json').read_text()):old_by[(r['event'],r['policy'])]=r
 for split in ['train','development']:
  meta=json.loads((CACHE/f'{split}_s{seed}/metadata.json').read_text());cd=CACHE/f'{split}_s{seed}'
  for ix,mm in enumerate(meta):
   d=data.load(mm['event']);rp,bp=model(d['s'],d['F']);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);g=d['g'];base={k:mm[k] for k in ['seed','event','split','trajectory','layer','head','chunk','step','tile']};ora=max(0.,float(g.max()));order=torch.argsort(sc,descending=True,stable=True);cache=np.load(cd/('event_%04d.npz'%mm['index']))
   for policy,scores in [('Set',sc),('Independent',ind)]:
    j=int(scores.argmax());ex=bool(scores[j]>0);gain=float(g[j]) if ex else 0.;newmask=d['mask'].clone()
    if ex:newmask[d['r'][j]]=False;newmask[d['a'][j]]=True
    assert int(newmask.sum())==int(d['mask'].sum()) and bool(newmask[d['own']])
    row=dict(**base,arm=arm,policy=policy,initial=d['initial'],oracle_gain=ora,executed_gain=gain,proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),execute=ex,action=j if ex else -1,proposal_index=j,predicted_gain=float(scores[j]),remove=int(d['r'][j]),add=int(d['a'][j]),top1=max(0.,float(g[order[0]])),top4=max(0.,float(g[order[:4]].max())),top8=max(0.,float(g[order[:8]].max())))
    rows.append(row)
    if arm=='T0' and split=='development':
     old=old_by[(mm['event'],policy)];assert row['action']==old['action'];delta=abs(row['executed_gain']-old['executed_gain']);assert delta<1e-9;replay.append(dict(event=mm['event'],policy=policy,gain_error=delta))
   err=(rp-d['rtrue']).square().mean(-1);be=bp-d['btrue'];pred=(d['lm']+bp).softmax(-1);under=d['mass']<d['alpha']/10;qs=data.qs.cpu().numpy()
   # Fixed E29 roles avoid post-training endpoint selection bias.
   for role in ['all','harmful_predicted_endpoint','oracle_endpoint','endpoint_pair','background']:
    reg=torch.as_tensor(cache['role_'+role],device='cuda')[None].expand_as(under)
    for status,sel in [('all',reg),('under10',reg&under),('other',reg&~under)]:
     if not bool(sel.any()):continue
     weights=d['alpha'][sel];weights=weights/weights.sum()
     info.append(dict(**base,arm=arm,role=role,status=status,query_pages=int(sel.sum()),local_mse=float(err[sel].mean()),true_mass_weighted_local_mse=float((weights*err[sel]).sum()),summary_weighted_local_sum=float((d['mass'].square()*err)[sel].sum()),beta_mae=float(be[sel].abs().mean()),beta_bias=float(be[sel].mean()),predicted_mass_share=float(pred[sel].sum()/16),true_mass_share=float(d['alpha'][sel].sum()/16),mass_l1=float((pred-d['alpha'])[sel].abs().sum()/16)))
     if arm=='T0':
      olderr=torch.as_tensor(cache['local_mse'][qs],device='cuda');oldbp=torch.as_tensor(cache['predicted_beta'][qs],device='cuda');olde=oldbp-d['btrue'];oldp=(d['lm']+oldbp).softmax(-1)
      info.append(dict(**base,arm='E29',role=role,status=status,query_pages=int(sel.sum()),local_mse=float(olderr[sel].mean()),true_mass_weighted_local_mse=float((weights*olderr[sel]).sum()),summary_weighted_local_sum=float((d['mass'].square()*olderr)[sel].sum()),beta_mae=float(olde[sel].abs().mean()),beta_bias=float(olde[sel].mean()),predicted_mass_share=float(oldp[sel].sum()/16),true_mass_share=float(d['alpha'][sel].sum()/16),mass_l1=float((oldp-d['alpha'])[sel].abs().sum()/16)))
   np.savez_compressed(dest/('%s_panel_%04d.npz'%(split,ix)),event=mm['event'],scores=sc.cpu().numpy(),independent=ind.cpu().numpy(),true_gain=g.cpu().numpy(),mask=d['mask'].cpu().numpy(),remove=d['r'].cpu().numpy(),add=d['a'].cpu().numpy())
   if ix%192==0:print('EVAL',seed,arm,split,ix,round(time.time()-start),flush=True)
  table(dest/'decisions.csv',rows);table(dest/'information.csv',info)
 data.check_frozen();table(dest/'T0_dev_replay.csv',replay);dump(dest/'evaluation_complete.json',dict(seed=seed,arm=arm,checkpoint_sha=sha(path),final_epoch=1,events=1440,dev_T0_replay=len(replay),test=False,raw_KV=False,teacher_in_decision=False,teacher_norm_for_offline_units_only=True,seconds=time.time()-start));print('EVAL DONE',seed,arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['T0','T1'],required=True);a=p.parse_args();evaluate(a.seed,a.arm)
