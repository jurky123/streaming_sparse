from .core import *

@torch.no_grad()
def runB(seed,limit=None):
 torch.set_num_threads(2);dest=DEST/f'B_s{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'complete.json').exists();bundle=bundle_for('C',seed);basis=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double();ortho=torch.linalg.qr(basis).Q;qs=torch.linspace(0,63,16,device='cuda').round().long();q64=torch.arange(64,device='cuda');rows=[];regions=[];rescues=[];ladders=[];accept=[];sources=[];start=time.time()
 for idx,path in enumerate(files('development')):
  if limit is not None and idx>=limit:break
  d,s,c,rt,bt,lm,cen=prepare_local(bundle,path);xp=E30/f's{seed}/beta_{idx:04d}.pt';pp=E30/f's{seed}/panel_{idx:04d}.npz';x=torch.load(xp,map_location='cuda',weights_only=True);p=np.load(pp);event=str(path.relative_to(OUT));assert event==x['event']==str(p['event']);assert torch.allclose(lm,x['logmass'],atol=1e-9,rtol=1e-9) and torch.allclose(cen,x['center'],atol=1e-8,rtol=1e-8) and torch.allclose(bt,x['beta_true'],atol=1e-8,rtol=1e-8);rp=x['predicted_response'].double();bp=x['betas']['Predicted'].double();mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');g=p['true_initial_gain'];norm=d['dense'].square().sum()/64+1e-12/64;J=float(objective(c,d['N'],d['Z'],d['dense'],mask));meta=dict(seed=seed,event=event,**{k:d['ev'][k] for k in ['trajectory','layer','chunk','step','head','tile']});j=pick(torch.as_tensor(p['initial_Predicted']));opt=int(np.argmax(g)) if g.max()>0 else -1;pair=[j,opt];mtrue=value(g,j)-value(g,opt);C=torch.zeros_like(mask);C[r]=True;C[a]=True;assert int(C.sum())==48
  response=[rp[qs],rt[qs]];betas=[bp[qs],bt[qs]];M={};vertices={}
  for ir,ib in [(0,0),(1,0),(0,1),(1,1)]:
   key=f'M{ir}{ib}';vv=pair_scores(response[ir],betas[ib],lm[qs],cen[qs],mask,r,a,pair,norm);M[key]=vv[0]-vv[1];vertices[key]=vv
  phir=.5*((M['M00']-M['M10'])+(M['M01']-M['M11']));phib=.5*((M['M00']-M['M01'])+(M['M10']-M['M11']));gamma=M['M11']-M['M10']-M['M01']+M['M00'];assert abs(phir+phib-(M['M00']-M['M11']))<1e-10
  endpoint=torch.zeros_like(mask);endpoint[endpoints(j,r,a)+endpoints(opt,r,a)]=True;parts=[endpoint,C&~endpoint,~C];assert torch.stack(parts).sum(0).eq(1).all()
  row=dict(**meta,initial=J,oracle_gain=max(0.,float(g.max())),j_pred=j,j_oracle=opt,same_action=j==opt,proposed_gain=value(g,j),M_true=mtrue,flip_error=M['M00']-mtrue,phi_response=phir,phi_beta=phib,interaction=gamma,remaining_teacher_error=M['M11']-mtrue,**M);rows.append(row);vreg=[]
  for bits in range(8):
   use=torch.zeros_like(mask)
   for k in range(3):
    if bits&(1<<k):use|=parts[k]
   rr=torch.where(use[None,:,None],rt[qs],rp[qs]);bb=torch.where(use[None],bt[qs],bp[qs]);vreg.append(margin(rr,bb,lm[qs],cen[qs],mask,r,a,pair,norm))
  sh=shapley3(vreg);assert abs(sum(sh)-(vreg[0]-vreg[7]))<1e-9;assert abs(vreg[0]-M['M00'])<1e-9 and abs(vreg[7]-M['M11'])<1e-9
  regions.append(dict(**meta,j_pred=j,j_oracle=opt,end_pages=int(endpoint.sum()),other_candidate_pages=int(parts[1].sum()),background_pages=int(parts[2].sum()),phi_endpoint=sh[0],phi_other=sh[1],phi_background=sh[2],**{f'M{k:03b}':v for k,v in enumerate(vreg)}))
  if j>=0 and g[j]<-1e-12:
   ownend=torch.zeros_like(mask);ownend[endpoints(j,r,a)]=True
   for mode in ['beta','response','both']:
    rr=torch.where(ownend[None,:,None],rt[qs],rp[qs]) if mode!='beta' else rp[qs];bb=torch.where(ownend[None],bt[qs],bp[qs]) if mode!='response' else bp[qs];vv=pair_scores(rr,bb,lm[qs],cen[qs],mask,r,a,[j,-1],norm)[0];out=dict(**meta,mode=mode,old_action=j,old_gain=float(g[j]),original_action_corrected_score=vv,original_action_rejected=vv<=0)
    if mode=='both':
     scores,_=local_scores(rr,bb,lm[qs],cen[qs],mask,r,a,norm);new=pick(scores);gain=value(g,new);out.update(new_action=new,new_gain=gain,result='noop' if new<0 else 'positive' if gain>1e-12 else 'harmful' if gain< -1e-12 else 'near_zero')
    rescues.append(out)
  tb=pick(torch.as_tensor(p['initial_True']))
  if tb>=0 and g[tb]<-1e-12:
   pairtb=[tb,opt];vv0=pair_scores(rp[qs],bt[qs],lm[qs],cen[qs],mask,r,a,pairtb,norm);vv1=pair_scores(rt[qs],bt[qs],lm[qs],cen[qs],mask,r,a,pairtb,norm);vv2=true_pair(c,d,mask,r,a,pairtb,q64,ortho,norm);vv3=true_pair(c,d,mask,r,a,pairtb,q64,None,norm);vvperp=true_pair(c,d,mask,r,a,pairtb,q64,ortho,norm,perp=True);assert max(abs(vv3[k]-vv2[k]-vvperp[k]) for k in range(2))<1e-8;assert max(abs(vv3[k]-value(g,jj)) for k,jj in enumerate(pairtb))<1e-7
   checkm16=true_pair(c,d,mask,r,a,pairtb,qs,basis,norm);assert max(abs(checkm16[k]-vv1[k]) for k in range(2))<1e-7
   ladders.append(dict(**meta,j_truebeta=tb,j_oracle=opt,**{f'{name}_{which}':vals[k] for name,vals in [('pred_M16',vv0),('true_M16_B16',vv1),('true_Q64_B16',vv2),('true_Q64_full',vv3),('true_Q64_perp',vvperp)] for k,which in enumerate(['wrong','oracle'])},projection_closure=max(abs(vv3[k]-vv2[k]-vvperp[k]) for k in range(2))))
  # Original highest proposal, regardless of original acceptance; gate may only refuse.
  proposal=int(np.argmax(p['initial_Predicted']));part=proposal_parts(rp[qs],bp[qs],lm[qs],cen[qs],mask,r,a,proposal,norm);assert part['necessary_condition'];accept.append(dict(**meta,initial=J,proposal=proposal,executed=j>=0,true_proposal_gain=float(g[proposal]),oracle_gain=max(0.,float(g.max())),**part))
  torch.save(dict(event=event,true_response=rt.cpu(),norm64=float(norm),initial=J,meta=meta,orthogonal_basis_error=float((basis.T@basis-torch.eye(16,device='cuda')).abs().max())),dest/f'target_{idx:04d}.pt');np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=event,vertices=np.array(list(vertices.values())),regions=np.array(vreg),pair=np.array(pair),true_gain=g)
  sources.append(dict(event=event,event_sha=sha(path),beta_sha=sha(xp),panel_sha=sha(pp),kv_file=str((path.parent/d['ev']['kv_file']).relative_to(OUT))))
  if (idx+1)%48==0:print('B',seed,idx+1,round(time.time()-start),flush=True)
 for name,data in [('action_margin_attribution',rows),('region_attribution',regions),('endpoint_rescue',rescues),('teacher_residual_ladder',ladders),('acceptance_features',accept),('sources',sources)]:dump(dest/(name+'.json'),data)
 if limit is None:dump(dest/'complete.json',dict(seed=seed,events=len(rows),protocol_sha=sha(DEST/'protocol.json'),C_sha=sha(checkpoint('C',seed)),reader_sha=sha(E29/f's{seed}/selected.pt'),phi_sha=sha(E27/f's{seed}/selected.pt'),basis_sha=sha(OUT/'fit.pt'),raw_kv_offline_reconstruction_authorized=True,training=False,test=False,seconds=time.time()-start))
 print('DONE B',seed,len(rows),flush=True)

@torch.no_grad()
def runA(seed,limit=None):
 torch.set_num_threads(2);assert all((DEST/f'B_s{s}/complete.json').exists() for s in [17,29]);dest=DEST/f'A_s{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'complete.json').exists();qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[];conditions=[];weights=[];qualities=[];start=time.time()
 for idx,path in enumerate(files('development')):
  if limit is not None and idx>=limit:break
  x=torch.load(E30/f's{seed}/beta_{idx:04d}.pt',map_location='cuda',weights_only=True);p=np.load(E30/f's{seed}/panel_{idx:04d}.npz');target=torch.load(DEST/f'B_s{seed}/target_{idx:04d}.pt',weights_only=True);meta=target['meta'];norm=target['norm64'];J=target['initial'];rp=x['predicted_response'][qs].double();bp=x['betas']['Predicted'][qs].double();bt=x['beta_true'][qs].double();lm=x['logmass'][qs];cen=x['center'][qs];mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');g=p['true_initial_gain'];C=torch.zeros_like(mask);C[r]=True;C[a]=True;ip={};cp={};indout={};jp=pick(torch.as_tensor(p['initial_Predicted']));jo=int(g.argmax()) if g.max()>0 else -1
  for method in METHODS:
   def getbeta(st,state):
    if method=='E29':return bp
    if method=='TrueBeta':return bt
    group,kind=method.split('-');bb,info=role_beta(rp,bp,bt,lm,cen,st,C,int(group[1:]),kind)
    for inf in info:weights.append(dict(**meta,method=method,state=state,**inf))
    return bb
   beta=getbeta(mask,0);sc,ind=local_scores(rp,beta,lm,cen,mask,r,a,norm);ip[method]=sc.cpu().numpy();indout[method]=ind.cpu().numpy()
   for policy,scores in [('Set',sc),('Independent',ind)]:
    j=pick(scores);prop=int(scores.argmax());gain=value(g,j);rows.append(dict(**meta,method=method,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),execute=j>=0,action=j,remove=int(r[j]) if j>=0 else -1,add=int(a[j]) if j>=0 else -1,proposal_gain=float(g[prop]),hindsight_gain=max(0.,float(g[prop])),executed_gain=gain,predicted_gain=value(scores,j),fixed_margin=value(scores,jp)-value(scores,jo)))
   order=np.argsort(-ip[method],kind='stable');qualities.append(dict(**meta,method=method,top1=max(0.,float(g[order[0]])),top8=max(0.,float(g[order[:8]].max())),oracle_gain=max(0.,float(g.max())),spearman=float(spearmanr(ip[method],g).statistic),sign_accuracy=float(((ip[method]>0)==(g>1e-12)).mean())))
   rr=torch.as_tensor(p['remove'],device='cuda');aa=torch.as_tensor(p['add'],device='cuda');common=[]
   for state,st in enumerate(p['states']):
    mm=torch.as_tensor(st,device='cuda');bb=beta if state==0 else getbeta(mm,state);ss,_=local_scores(rp,bb,lm,cen,mm,rr,aa,norm);common.append(ss.cpu().numpy())
   cp[method]=np.stack(common);conditions.append(dict(**meta,method=method,**conditional_metrics(p['true_gain'],cp[method])))
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=x['event'],true_gain=p['true_gain'],true_initial_gain=g,states=p['states'],remove=p['remove'],add=p['add'],initial_remove=p['initial_remove'],initial_add=p['initial_add'],**cp,**{'initial_'+k:v for k,v in ip.items()},**{'independent_'+k:v for k,v in indout.items()})
  if (idx+1)%48==0:print('A',seed,idx+1,round(time.time()-start),flush=True)
 for name,data in [('decisions',rows),('conditional',conditions),('weights',weights),('quality',qualities)]:
  for row in data:
   for k,v in row.items():
    if isinstance(v,float) and not math.isfinite(v):row[k]=None
  dump(dest/(name+'.json'),data)
 if limit is None:dump(dest/'complete.json',dict(seed=seed,events=288,protocol_sha=sha(DEST/'protocol.json'),training=False,test=False,raw_kv_loaded=False,seconds=time.time()-start))
 print('DONE A',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['A','B'],required=True);p.add_argument('--limit',type=int);args=p.parse_args();{'A':runA,'B':runB}[args.stage](args.seed,args.limit)
