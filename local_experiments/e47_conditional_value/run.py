"""E43 frozen evidence completion; no new fitting or checkpoint selection."""
from ..e44_fit_diagnostics.core import *
from ..e41_training_signal.report import decision_summary
E44=D;D=ROOT/'artifacts/e47_conditional_value_v1'

def run(seed):
 torch.set_num_threads(2);dest=D/f's{seed}';dest.mkdir(parents=True,exist_ok=True)
 dump(dest/'protocol.json',dict(seed=seed,models='P=E44 P_set final; L=original E43 reference; P_to_L transplant of E44 P_set',training=False,Test=False,heldout='one new background state per original event: second eligible selected and second eligible unselected outside initial Medium; own protected; no label-based selection',comparisons=['current-state scores','other-trained-state preference','P-to-L transplant'],queries='original Q64, M16 probes; heldout states only, no new Query claim',units='same canonical Q64 full target',initialization='frozen existing models',checkpoint_selection=False))
 data,states,groups,initial=build(seed,verify=True);P=model_from(E44/f's{seed}/P_set_final.pt').eval();L=Lookup(groups,initial).cuda();old=torch.load(E43/f's{seed}/L_final.pt',map_location='cuda',weights_only=True)['co']
 for gid,ids in groups.items():L.co[L.group_index[gid]].data.copy_(old[ids])
 transplant=Lookup(groups,initial).cuda()
 with torch.no_grad():
  for gid in groups:
   d=next(x for x in states if x['group']==gid);transplant.co[transplant.group_index[gid]].copy_(P.page(page_features(d['s'])))
 models={'P':P,'L':L,'P_to_L':transplant};rows=[];checks=[];arrays={};condition=[]
 with torch.no_grad():
  for eventidx in range(48):
   pair=states[2*eventidx:2*eventidx+2];base=pair[0];C=torch.zeros_like(base['mask']);C[base['r']]=True;C[base['a']]=True;rr=torch.where(base['mask']&~C)[0];rr=rr[rr!=base['own']];aa=torch.where(~base['mask']&~C)[0];assert len(rr)>=2 and len(aa)>=2;mask=base['mask'].clone();mask[rr[1]]=False;mask[aa[1]]=True
   held=data.load(base['meta']['event'],dict(mask=mask.cpu().tolist(),remove=base['r'].cpu().tolist(),add=base['a'].cpu().tolist()));held.update(ids=base['ids'],group=base['group']);assert not torch.equal(mask,pair[0]['mask']) and not torch.equal(mask,pair[1]['mask']);allstates=pair+[held];scores={};inds={};outputs_all={}
   for name,model in models.items():
    ss=[];ii=[];oo=[]
    for d in allstates:
     rp,bp=outputs(model,d);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);ss.append(sc);ii.append(ind);oo.append((rp,bp))
    scores[name]=ss;inds[name]=ii;outputs_all[name]=oo
    for stateidx,d in enumerate(allstates):
     meta=d['meta'];role=['Risk','BgSwap_train','BgSwap_holdout'][stateidx];rp,bp=oo[stateidx]
     for policy,score,source in [('Set',ss[stateidx],stateidx),('Independent',ii[stateidx],stateidx),('Reuse_other',ss[1-stateidx] if stateidx<2 else ss[0],1-stateidx if stateidx<2 else 0)]:
      j=int(score.argmax());ex=bool(score[j]>0);nm=d['mask'].clone()
      if ex:nm[d['r'][j]]=False;nm[d['a'][j]]=True
      assert int(nm.sum())==int(d['mask'].sum()) and bool(nm[d['own']]);gain=float(d['g'][j]) if ex else 0.
      rows.append(dict(seed=seed,event=meta['event'],event_index=eventidx,layer=meta['layer'],trajectory=meta['trajectory'],head=meta['head'],state_role=role,arm=name,policy=policy,preference_source=source,action=j if ex else -1,proposal_index=j,remove=int(d['r'][j]) if ex else -1,add=int(d['a'][j]) if ex else -1,execute=ex,initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=gain,proposal_gain=float(d['g'][j]),hindsight_gain=max(0.,float(d['g'][j])),local_mse=float((rp-d['rtrue']).square().mean()),beta_mae=float((bp-d['btrue']).abs().mean())))
    for i in range(3):arrays[f'{eventidx}_{name}_{i}']=ss[i].cpu().numpy();arrays[f'{eventidx}_{name}_ind_{i}']=ii[i].cpu().numpy()
    condition.append(dict(seed=seed,event=base['meta']['event'],arm=name,**conditional_metrics(np.stack([d['g'].cpu().numpy() for d in pair]),np.stack([x.cpu().numpy() for x in ss[:2]]))))
   for i,d in enumerate(allstates):
    for k in range(2):assert torch.equal(outputs_all['P'][i][k],outputs_all['P_to_L'][i][k])
    assert torch.equal(scores['P'][i],scores['P_to_L'][i]);checks.append(dict(event=d['meta']['event'],state=i,coefficient_outputs_exact=True,scores_exact=True))
    for n,val in [('truth',d['g']),('remove',d['r']),('add',d['a']),('mask',d['mask'])]:arrays[f'{eventidx}_{n}_{i}']=val.cpu().numpy()
 table(dest/'decisions.csv',rows);table(dest/'transplant_checks.csv',checks);table(dest/'conditional.csv',condition);np.savez_compressed(dest/'scores.npz',**arrays)
 # Exact action and gain replay versus E44 saved original E43 matrices.
 x=np.load(E44/f's{seed}/scores_-1.npz');xp=np.load(E44/f's{seed}/scores_2304.npz');maxerr=0.
 for e in range(48):
  for i in range(2):
   for name in ['P','L']:
    err=float(np.max(abs(arrays[f'{e}_{name}_{i}']-(xp['P_set'] if name=='P' else x['E43_L'])[2*e+i])));assert err==0;maxerr=max(maxerr,err)
 data.check_frozen();dump(dest/'audit.json',dict(passed=True,events=48,train_states=96,heldout_states=48,P_to_L_exact_states=144,E44_Pset_and_E43_L_matrix_replay_exact=True,max_score_difference=maxerr,identity_audit=str(E44/f's{seed}/identity_audit.json'),raw_attention=False,training=False,Test=False));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
