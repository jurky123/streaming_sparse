from ..e46_supervision_bridge.core import *
D=ROOT/'artifacts/e48_tail_state_v1'

@torch.no_grad()
def run(seed,phase):
 torch.set_num_threads(2);data=Data(seed);dest=D/f's{seed}';names=['Full_Pset'];models={}
 for name in names:
  if name=='Layer_split':m=LayerReader(seed);m.load_state_dict(torch.load(dest/'layer_final.pt',map_location='cuda',weights_only=True))
  else:m=model_from(E43/f's{seed}/P_final.pt' if name=='E43_P' else E44/f's{seed}/P_set_final.pt' if name=='E44_Pset' else dest/'fullset_final.pt')
  models[name]=m.eval()
 chosen=set(json.loads((E43/f's{seed}/protocol.json').read_text())['events']);chosen_m=[data.meta[e] for e in chosen];trainheads={(m['trajectory'],m['layer']):m['head'] for m in chosen_m};rows=[];info=[];audit=[];start=time.time()
 for split in ['train','development']:
  metas=json.loads((CACHE/f'{split}_s{seed}/metadata.json').read_text())
  for i,meta in enumerate(metas):
   d=data.load(meta['event']);base={k:meta[k] for k in ['seed','event','split','trajectory','layer','head','chunk','step','tile']}
   cohort='dev_unseen_trajectory' if split=='development' else 'fitted48' if meta['event'] in chosen else 'same_chunk_step_otherhead' if meta['chunk']==5 and meta['step']==0 else 'same_head_othercontext' if meta['head']==trainheads[(meta['trajectory'],meta['layer'])] else 'other_head_othercontext'
   for name,model in models.items():
    m=model.active(d) if isinstance(model,LayerReader) else model;rp,bp=m(d['s'],d['F']);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);order=torch.argsort(sc,descending=True,stable=True)
    for policy,score in [('Set',sc),('Independent',ind)]:
     j=int(score.argmax());ex=bool(score[j]>0);mask=d['mask'].clone()
     if ex:mask[d['r'][j]]=False;mask[d['a'][j]]=True
     assert bool(mask[d['own']]) and int(mask.sum())==int(d['mask'].sum())
     rows.append(dict(**base,cohort=cohort,arm=name,policy=policy,initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=float(d['g'][j]) if ex else 0.,execute=ex,action=j if ex else -1,remove=int(d['r'][j]) if ex else -1,add=int(d['a'][j]) if ex else -1,proposal_gain=float(d['g'][j]),hindsight_gain=max(0.,float(d['g'][j])),top8=max(0.,float(d['g'][order[:8]].max()))))
    info.append(dict(**base,cohort=cohort,arm=name,local_mse=float((rp-d['rtrue']).square().mean()),local_mass_mse=float((d['mass'][...,None]*(rp-d['rtrue'])).square().mean()),beta_mae=float((bp-d['btrue']).abs().mean())))
    if i%192==0:
     alt,_=safe_predictions(reconstruct(rp,bp,d['lm'],d['center']),bp,d['lm'],d['mask'],d['r'],d['a'],d['norm']);err=float((alt-sc).abs().max());assert torch.allclose(alt,sc,rtol=1e-5,atol=1e-8);audit.append(dict(event=meta['event'],arm=name,score_error=err))
   if i%192==0:print('EVAL',seed,phase,split,i,round(time.time()-start),flush=True)
  table(dest/f'{phase}_decisions.csv',rows);table(dest/f'{phase}_information.csv',info)
 data.check_frozen();table(dest/f'{phase}_checks.csv',audit);dump(dest/f'{phase}_evaluation_complete.json',dict(seed=seed,phase=phase,events=1440,arms=names,Test=False,training=False,rawKV=False));print('EVAL DONE',seed,phase,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['frozen','trained'],required=True);a=p.parse_args();run(a.seed,a.phase)
