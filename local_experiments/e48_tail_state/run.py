from ..e46_supervision_bridge.core import *
D=ROOT/'artifacts/e48_tail_state_v1'
def run(seed):
 torch.set_num_threads(2);dest=D/f's{seed}';dest.mkdir(parents=True,exist_ok=True);data=Data(seed);events=['data/forest_run_s702/c8_l29_s0_h9_t39.pt','data/reef_fish_s703/c5_l29_s0_h9_t37.pt'];dump(dest/'protocol.json',dict(seed=seed,tail_events=events,states='all48 E43 train events; Risk plus deterministic seed4800+event-index permutations of selected/unselected non-own pages, swap counts 4/8/16; no true gain selection',training=False,Test=False,rawKV='tail logit concentration only',boundary='New representative-state pilot, not full-distribution pure-set training'))
 tails=[]
 for event in events:
  d=data.load(event);ev=torch.load(OUT/event,map_location='cpu',weights_only=True);sx=torch.load(CACHE/f'train_s{seed}'/d['meta']['summary_file'],map_location='cpu',weights_only=True);kv=torch.load(OUT/sx['kv_file'],map_location='cpu',weights_only=True);K=kv['k'].cuda().double();q=ev['q'].cuda().double();logits=torch.einsum('qd,ptd->qpt',q,K)/128**.5;token=logits.softmax(-1);truepage=ev['page_logz'].cuda().double().softmax(-1);model=model_from(E29/f's{seed}/selected.pt');rp,bp=model(d['s'],d['F']);sc,_=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);j=int(sc.argmax())
  for p,role in [(int(d['r'][j]),'remove'),(int(d['a'][j]),'add')]:
   for qi in range(64):
    rep=(q[qi]@d['s'].keys[0,p].double().T)/128**.5;bet=float(ev['page_logz'][qi,p].cuda()-torch.logsumexp(rep,0)-np.log(16));tails.append(dict(seed=seed,event=event,page=p,role=role,query=qi,true_mass=float(truepage[qi,p]),beta=bet,top1_token_mass=float(token[qi,p].max()),top2_token_mass=float(token[qi,p].topk(2).values.sum()),max_raw_logit=float(logits[qi,p].max()),max_rep_logit=float(rep.max()),initial=d['initial'],selected_action=j,true_gain=float(d['g'][j]),predicted_gain=float(sc[j])))
 table(dest/'tail.csv',tails)
 chosen=json.loads((E43/f's{seed}/protocol.json').read_text())['events'];rows=[];space=[];matrices={};model=model_from(ROOT/f'artifacts/e44_fit_diagnostics_v1/s{seed}/P_set_final.pt')
 for ei,event in enumerate(chosen):
  base=data.load(event);P=base['mask'].numel();rng=np.random.default_rng(4800+ei);sel=torch.where(base['mask'])[0].cpu().numpy();sel=sel[sel!=base['own']];uns=torch.where(~base['mask'])[0].cpu().numpy();sel=rng.permutation(sel);uns=rng.permutation(uns);states=[base]
  for n in [4,8,16]:
   n=min(n,len(sel),len(uns));mask=base['mask'].clone();mask[torch.as_tensor(sel[:n],device='cuda')]=False;mask[torch.as_tensor(uns[:n],device='cuda')]=True
   # Keep a common action panel legal by choosing endpoint pairs in the shared membership intersections.
   states.append(dict(mask=mask))
  commonR=base['mask'].clone();commonA=~base['mask']
  for st in states:commonR&=st['mask'];commonA&=~st['mask']
  commonR[base['own']]=False;r=torch.where(commonR)[0][:16];a=torch.where(commonA)[0][:32];assert len(r) and len(a);rr=r.repeat_interleave(len(a));aa=a.repeat(len(r));truth=[];pred=[]
  for si,st in enumerate(states):
   d=data.load(event,dict(mask=st['mask'].cpu().tolist(),remove=rr.cpu().tolist(),add=aa.cpu().tolist()));rp,bp=model(d['s'],d['F']);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);truth.append(d['g'].cpu().numpy());pred.append(sc.detach().cpu().numpy());j=int(sc.argmax());rows.append(dict(seed=seed,event=event,state=si,actions=len(rr),initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=float(d['g'][j]) if sc[j]>0 else 0.,execute=bool(sc[j]>0)))
  t=np.stack(truth);ad=float(np.maximum(0,t.max(1)).mean());fx=float(np.maximum(0,t).mean(0).max());space.append(dict(seed=seed,event=event,adaptive=ad,fixedplus=fx,conditional=ad-fx));matrices[str(ei)+'_true']=t;matrices[str(ei)+'_pred']=np.stack(pred)
 table(dest/'states.csv',rows);table(dest/'conditional.csv',space);np.savez_compressed(dest/'scores.npz',**matrices);dump(dest/'complete.json',dict(seed=seed,states=len(rows),tail_rows=len(tails),training=False,Test=False));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
