"""E33-A: held-context page-local mass fitting; no network training."""
from ..e31_beta_reuse.run import *
DEST=ROOT/'artifacts/e33_mass_representatives_v1'
KINDS=['tile','step','forward'];METHODS=['A1','A2']

def fit_pages(q,y,k0,kscale,method,steps=200):
 # q contains donor queries only; y donor exact raw page logZ. Batch pages independent.
 with torch.enable_grad():
  par=torch.zeros(k0.shape if method=='A2' else k0.shape[:2],device=k0.device,dtype=torch.float64,requires_grad=True);opt=torch.optim.Adam([par],lr=.02 if method=='A2' else .05)
  def calc():
   keys=k0+par*kscale[:,None,None] if method=='A2' else k0
   lp=par.log_softmax(-1) if method=='A1' else torch.full(k0.shape[:2],-math.log(4),device=k0.device,dtype=k0.dtype)
   z=(torch.einsum('qd,pgd->qpg',q,keys)/math.sqrt(128)+lp[None]+math.log(64)).logsumexp(-1)
   mse=(z-y).square().mean(0)
   reg=(par-par.mean(-1,keepdim=True)).square().mean(-1)*1e-6 if method=='A1' else par.square().mean((1,2))*1e-4
   return mse+reg,mse,keys,lp
  with torch.no_grad():loss,mse,_,_=calc();initial=mse.clone();best=loss.clone();bestpar=par.clone();bestm=mse.clone();mid=None
  for step in range(steps):
   opt.zero_grad();loss,mse,_,_=calc();loss.sum().backward();opt.step()
   with torch.no_grad():
    loss,mse,_,_=calc();better=loss<best;bestpar=torch.where(better.reshape((-1,)+(1,)*(par.ndim-1)),par,bestpar);bestm=torch.where(better,mse,bestm);best=torch.minimum(best,loss)
    if step==steps-21:mid=bestm.clone()
  with torch.no_grad():
   par.copy_(bestpar);_,_,keys,lp=calc();keys=keys.detach().clone();lp=lp.detach().clone();qk=keys.bfloat16().double();pi=lp.exp().bfloat16().double();assert (pi>0).all();qlp=(pi/pi.sum(-1,keepdim=True)).log();z=(torch.einsum('qd,pgd->qpg',q,qk)/math.sqrt(128)+qlp[None]+math.log(64)).logsumexp(-1);quant=(z-y).square().mean(0)
   info=dict(pages=len(k0),donor_queries=len(q),initial_mse=float(initial.mean()),fitted_mse=float(bestm.mean()),quantized_mse=float(quant.mean()),last20_improvement=float((mid-bestm).mean()),max_key_shift=float((keys-k0).abs().max()),weight_min=float(qlp.exp().min()),weight_max=float(qlp.exp().max()),steps=steps)
 return qk,qlp,info

def weighted_contributions(d,s,m,keys,lp):
 q=d['q'].double();nu=s.mean[0].double()[:,None]+torch.einsum('ag,pak,dk->pgd',haar(q).double(),s.detail[0].double(),m.basis[0].bfloat16().double());logit=torch.einsum('qd,pgd->qpg',q,keys)/math.sqrt(128)+lp[None]+math.log(64);w=(logit-d['log_scale'][:,None,None]).exp();zh=w.sum(-1);nh=torch.einsum('qpg,pgd->qpd',w,nu);A=zh.sum(1);o=nh.sum(1)/A[:,None];return dict(nh=nh,zh=zh,A=A,o=o,mass=zh/A[:,None],logit=logit,logzh=logit.logsumexp(-1)-d['log_scale'][:,None],nu=nu)

def weighted_route(d,s,m,keys,lp):
 q=d['q'][torch.linspace(0,63,16,device='cuda').round().long()].double();st=s.stats[0].double().clone();km=d['k'][0].double().reshape(-1,4,16,128).mean(2);changed=(keys!=s.keys[0].double()).any((1,2));st[changed,:,8]=(km-keys).norm(dim=-1).bfloat16().double()[changed];z=torch.einsum('qd,pgd->qpg',q,keys)/math.sqrt(128)+lp[None];prob=z.flatten(1).softmax(-1).reshape_as(z);alpha=prob.sum(-1);q2=q.square().reshape(16,8,16).sum(-1)/128;qn=q.norm(dim=-1)/math.sqrt(128);var=torch.einsum('qj,pgj->qpg',q2,st[:,:,:8]).clamp_min(0);bb=qn[:,None,None]*st[None,:,:,8];bp=st[:,1,11];cc=bp+bp.max();x0=alpha*st[None,:,0,11]+((prob-alpha[...,None]/4).abs()*st[None,:,:,9]).sum(-1);x1=(prob*var.sqrt()*st[None,:,:,10]).sum(-1);x2=cc[None]*(prob*bb).sum(-1);x3=cc[None]*(prob*(var+bb.square())).sum(-1);risk=(torch.stack([x0,x1,x2,x3],-1)*m.theta[0].double()).sum(-1).mean(0);mask=RouteOps.select(risk[None,None],torch.tensor([[d['own']]],device='cuda'),Config())['mask'][0,0];r,a=pool({'risk':risk.cpu().numpy()},mask.cpu().numpy(),d['own']);return mask,torch.as_tensor(r,device='cuda'),torch.as_tensor(a,device='cuda'),risk

def decomposition(d,c,mask):
 # Dense-normalized V and Z error sources, using canonical stable page output.
 od=d['dense'];zd=d['Z'].sum(1);op=d['ev']['page_output'].cuda().double();oh=torch.einsum('qpg,pgd->qpd',c['logit'].softmax(-1),c['nu']);ez=(d['Z']-c['zh'])[...,None]/zd[:,None,None]*(oh-od[:,None]);ev=d['Z'][...,None]/zd[:,None,None]*(op-oh);V=ev[:,~mask].sum(1);Z=ez[:,~mask].sum(1);den=torch.where(mask[None],d['Z'],c['zh']).sum(1)/zd;norm=od.square().sum()+1e-12;v=(V/den[:,None]).square().sum()/norm;z=(Z/den[:,None]).square().sum()/norm;cross=2*(V*Z/den[:,None].square()).sum()/norm;J=objective(c,d['N'],d['Z'],od,mask);err=float(abs(v+z+cross-J));assert err<1e-7+1e-7*float(J),(err,float(J));return dict(V_energy=float(v),Z_energy=float(z),cross_energy=float(cross),net_nmse=float(J),identity_error=err)

def run(seed,limit=None):
 torch.set_num_threads(2);dest=DEST/f's{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'complete.json').exists();bundle=bundle_for('C',seed);paths=files('development');groups=collections.defaultdict(list);ident=json.loads((ROOT/f'artifacts/e28_local_response_v1/s{seed}/identities.json').read_text());ig=collections.defaultdict(list)
 for x in ident:ig[tuple(x['group'])].append(x)
 for idx,path in enumerate(paths):
  ev=torch.load(path,map_location='cpu',weights_only=True)
  if ev['head']==0:groups[(ev['trajectory'],ev['layer'],ev['head'])].append((idx,path))
 rows=[];fits=[];supports=[];audit=[];start=time.time();count=0
 for gi,(group,items) in enumerate(groups.items()):
  if limit is not None and gi>=limit:break
  records=[];emap={}
  with torch.no_grad():
   for idx,path in items:
    d,s,c,_,_,_,_=prepare_local(bundle,path);k=s.keys[0].double();lp=torch.full(k.shape[:2],-math.log(4),device='cuda',dtype=torch.float64);m=bundle.layers[d['key']];base=weighted_contributions(d,s,m,k,lp);mask,r,a=route(d['q'][torch.linspace(0,63,16,device='cuda').round().long()],s,m,d['own']);mr,rr,aa,_=weighted_route(d,s,m,k,lp)
    # Risk reference stats already rounded; A2 refresh only when Keys actually change.
    emap[str(path.relative_to(OUT))]=len(records);rec=dict(idx=idx,path=path,ev=d['ev'],event=str(path.relative_to(OUT)),d=d,s=s,base=base,k=k,lp=lp,mask=mask,r=r,a=a,provenance=[],support={z:torch.zeros(len(k),device='cuda',dtype=torch.bool) for z in KINDS},variants={})
    for method in METHODS:
     for kind in KINDS+['matched-tile','matched-step']:rec['variants'][method+'-'+kind]=[k.clone(),lp.clone()]
    rec['summary_sha']=[digest(*(getattr(s,n)[0,p] for n in ['keys','mean','detail','stats'])) for p in range(d['hp'])];records.append(rec);audit.append(dict(seed=seed,event=rec['event'],base_contribution_error=float((base['nh']-c['nh']).abs().max()),risk_float64_same_mask=bool(torch.equal(mask,mr))))
  sigs=collections.defaultdict(list)
  for x in ig[group]:
   occ=[(emap[o['event']],o['page']) for o in x['occurrences']]
   for i,p in occ:assert records[i]['summary_sha'][p]==x['summary_sha']
   if len(occ)>1:sigs[tuple(i for i,p in occ)].append(occ)
  for si,(sig,occ) in enumerate(sigs.items()):
   ix=[torch.tensor([o[j][1] for o in occ],device='cuda') for j in range(len(sig))];cache={}
   for j,(i,pp) in enumerate(zip(sig,ix)):
    rec=records[i];dd={kind:donors(records,sig,j,kind) for kind in KINDS};matched=all(dd.values())
    for kind,kk in dd.items():
     if not kk:continue
     rec['support'][kind][pp]=True
     assert j not in kk
     rec['provenance'].append(dict(kind=kind,pages=pp.cpu().tolist(),donor_events=[records[sig[t]]['event'] for t in kk],donor_pages=[ix[t].cpu().tolist() for t in kk]))
     for method in METHODS:
      key=(method,tuple(kk))
      if key not in cache:
       q=torch.cat([records[sig[t]]['d']['q'].double() for t in kk]);y=torch.cat([records[sig[t]]['d']['ev']['page_logz'].cuda().double()[:,ix[t]] for t in kk]);donor=records[sig[kk[0]]];dp=ix[kk[0]];k0=donor['k'][dp];raw=donor['d']['k'][0,dp].double().reshape(-1,4,16,128);scale=(raw-raw.mean(2,keepdim=True)).square().mean((1,2,3)).sqrt().clamp_min(1e-3);K,L,info=fit_pages(q,y,k0,scale,method);cache[key]=(K,L);fits.append(dict(seed=seed,group=str(group),signature=si,method=method,donor_events=[records[sig[t]]['event'] for t in kk],**info))
      K,L=cache[key];rec['variants'][method+'-'+kind][0][pp]=K;rec['variants'][method+'-'+kind][1][pp]=L
      if matched and kind in ['tile','step']:rec['variants'][method+'-matched-'+kind][0][pp]=K;rec['variants'][method+'-matched-'+kind][1][pp]=L
  with torch.no_grad():
   for rec in records:
    d,s=rec['d'],rec['s'];m=bundle.layers[d['key']];old=rec['mask'];base=rec['base'];J0=float(objective(base,d['N'],d['Z'],d['dense'],old));meta=dict(seed=seed,event=rec['event'],**{k:rec['ev'][k] for k in ['trajectory','layer','head','chunk','step','tile']});saved={};allvariants={'Baseline':(rec['k'],rec['lp']),**rec['variants']}
    for name,(keys,lp) in allvariants.items():
     c=weighted_contributions(d,s,m,keys,lp);same=float(objective(c,d['N'],d['Z'],d['dense'],old))
     if name=='Baseline':mask,r,a=old,rec['r'],rec['a'];support=torch.arange(len(old),device='cuda')<d['hp']
     else:
      kind=name.split('-',1)[1];support=rec['support']['forward']&rec['support']['tile']&rec['support']['step'] if kind.startswith('matched') else rec['support'][kind];mask,r,a,_=weighted_route(d,s,m,keys,lp)
     J=float(objective(c,d['N'],d['Z'],d['dense'],mask));g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);best=int(g.argmax());gain=max(0.,float(g[best]));assert int(mask.sum())==int(old.sum()) and mask[d['own']]
     bt=d['logZ']-base['logzh'];beta=d['logZ']-c['logzh'];sel=beta[:,support];bsel=bt[:,support];mass=d['Z']/d['Z'].sum(1,keepdim=True);coverage=float(mass[:,support].sum()/mass[:,:d['hp']].sum());w=(d['logZ'][:,support]).flatten().softmax(0).reshape_as(sel) if sel.numel() else sel;delta=(c['logzh']-base['logzh'])[:,support]
     if name.startswith('A1'):assert float((c['logzh']-base['logzh']).max())<=math.log(4)+1e-9
     row=dict(**meta,method=name,covered_pages=int(support.sum()),history_pages=d['hp'],history_mass_fraction=coverage,old_risk=J0,same_mask_nmse=same,new_risk=J,new_oracle=J-gain,oracle_gain=gain,oracle_remove=int(r[best]),oracle_add=int(a[best]),mask_changes=int((mask!=old).sum())//2,beta_mae=float(sel.abs().mean()) if sel.numel() else None,beta_mse=float(sel.square().mean()) if sel.numel() else None,baseline_beta_mse=float(bsel.square().mean()) if sel.numel() else None,baseline_beta_mae=float(bsel.abs().mean()) if sel.numel() else None,beta_positive_gt_log4=float((bsel>math.log(4)).double().mean()) if sel.numel() else None,quality_relative_l1=float((w*torch.expm1(-sel).abs()).sum()) if sel.numel() else None,**decomposition(d,c,mask));rows.append(row);saved[name]=dict(keys=keys.cpu(),logpi=lp.cpu(),support=support.cpu(),mask=mask.cpu(),remove=r.cpu(),add=a.cpu(),gains=g.cpu())
    torch.save(dict(event=rec['event'],variants=saved,provenance=rec['provenance'],supports={k:v.cpu() for k,v in rec['support'].items()}),dest/f"event_{rec['idx']:04d}.pt");count+=1
  dump(dest/'rows.json',rows);dump(dest/'fits.json',fits);dump(dest/'audit.json',audit);dump(dest/'status.json',dict(groups=gi+1,events=count,seconds=time.time()-start));print('GROUP',seed,gi+1,count,round(time.time()-start),flush=True)
 if limit is None:
  assert count==72;dump(dest/'complete.json',dict(seed=seed,events=count,seconds=time.time()-start,network_training=False,test_accessed=False,protocol_sha=sha(DEST/'protocol.json'),C_sha=sha(checkpoint('C',seed))))
 print('DONE',seed,count,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit-groups',type=int);a=p.parse_args();run(a.seed,a.limit_groups)
