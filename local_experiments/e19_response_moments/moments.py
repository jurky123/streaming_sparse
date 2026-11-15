from .core import *
import argparse,torch,math

def array(t):return t.detach().cpu().numpy()
def gain_scores(e,zp,zh,mask,r,a,queries,details=False):
 device=e.device;ix=torch.as_tensor(mask,device=device);r=torch.as_tensor(r,device=device);a=torch.as_tensor(a,device=device);qs=torch.as_tensor(queries,device=device);E=e[:,~ix].sum(1);z=torch.where(ix[None],zp,zh).sum(1);dz=zp-zh;baseline=E.square().sum(-1)/z.square();out=[];pi=[];pb=[];bad=0
 if bool((z[qs]<=0).any()):return (None,len(r),None) if details else (None,len(r))
 for start in range(0,len(r),128):
  rr,aa=r[start:start+128],a[start:start+128];de=e[:,rr]-e[:,aa];en=E[:,None]+de;zn=z[:,None]-dz[:,rr]+dz[:,aa];invalid=(zn[qs]<=0).any(0);bad+=int(invalid.sum().item());out.append((baseline[:,None]-en.square().sum(-1)/zn.square())[qs].mean(0))
  if details:
   pi.append((-2*torch.einsum('qd,qad->qa',E,de)/z[:,None].square())[qs].mean(0));pb.append((de.square().sum(-1)/z[:,None].square())[qs].mean(0))
 score=torch.cat(out)
 if bad:return (None,bad,None) if details else (None,bad)
 assert bool(torch.isfinite(score).all())
 return (array(score),0,dict(I=array(torch.cat(pi)),B=array(torch.cat(pb)))) if details else (array(score),0)

def run(eid,device):
 out=OUT/'moments'/eid;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 path=DATA/'representatives'/f'{eid}.pt';snap=torch.load(path,weights_only=True,map_location='cpu');f=dict(np.load(DATA/'contributions'/eid/'summary_and_route_features.npz'));raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));st=np.load(E13/'events'/eid/'states.npz');an=np.load(E13/'events'/eid/'anchors.npz');ident=json.loads((DATA/'contributions'/eid/'identity.json').read_text())
 assert np.array_equal(snap['ids'].numpy(),f['token_ids']);assert np.unique(f['token_ids']).size==f['token_ids'].size;assert np.array_equal(snap['q'].float().numpy(),f['query'])
 for n in ['keys','mean','detail','stats']:assert np.array_equal(snap['summary'][n][0].float().numpy(),f['summary_'+n])
 assert np.array_equal(snap['basis'][0].bfloat16().float().numpy(),f['basis'])
 def T(x):return x.to(device=device,dtype=torch.float64) if isinstance(x,torch.Tensor) else torch.as_tensor(x,device=device,dtype=torch.float64)
 q=T(snap['q']);k=T(snap['k']);v=T(snap['v']);P=k.shape[0];kg=k.reshape(P,4,16,128);vg=v.reshape(P,4,16,128);kap=T(f['summary_keys']);basis=T(f['basis']);mu=T(f['summary_mean']);detail=T(f['summary_detail']);haar=T([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])
 nu=mu[:,None]+torch.einsum('ag,pak,dk->pgd',haar,detail,basis);ell=torch.einsum('qd,pgd->qpg',q,kap)/math.sqrt(128);logscale=(ell+math.log(16)).flatten(1).logsumexp(1);w=16*torch.exp(ell-logscale[:,None,None]);delta=torch.einsum('qd,pgtd->qpgt',q,kg-kap[:,:,None])/math.sqrt(128)
 zm=w.sum(-1);nm=torch.einsum('qpg,pgd->qpd',w,nu);ohat=nm.sum(1);dm=delta.mean(-1);d2=delta.square().mean(-1);vm=vg.mean(2);km=kg.mean(2);dv=torch.einsum('qpgt,pgtd->qpgd',delta,vg)/16;d2v=torch.einsum('qpgt,pgtd->qpgd',delta.square(),vg)/16
 cov=torch.einsum('pgtv,pgtk->pgvk',vg-vm[:,:,None],kg-km[:,:,None])/16;covq=torch.einsum('pgvk,qk->qpgv',cov,q)/math.sqrt(128);cov_err=float((dv-dm[...,None]*vm[None]-covq).abs().max().item());assert cov_err<1e-10
 # Explicit full second-moment tensors only for audit groups; all contractions elsewhere retain full order.
 second_error=0.
 for p,g in [(0,0),(P-1,3)]:
  dk=kg[p,g]-kap[p,g];mkk=dk.T@dk/16;mkkv=torch.einsum('ti,tj,tk->ijk',dk,dk,vg[p,g])/16;query=q[17]/math.sqrt(128);v2=torch.einsum('i,j,ijk->k',query,query,mkkv);z2=query@mkk@query;second_error=max(second_error,float((v2-d2v[17,p,g]).abs().max().item()),float(abs(z2-d2[17,p,g]).item()));del mkkv
 assert second_error<1e-9
 z0g=w*(1+dm);n0g=z0g[...,None]*vm[None];z1g=z0g;n1g=w[...,None]*(vm[None]+dv);z2g=w*(1+dm+.5*d2);n2g=w[...,None]*(vm[None]+dv+.5*d2v)
 expdelta=delta.exp();zeg=w*expdelta.mean(-1);neg=w[...,None]*torch.einsum('qpgt,pgtd->qpgd',expdelta,vg)/16;ze=zeg.sum(-1);ne=neg.sum(2);total=ze.sum(1);od=ne.sum(1)/total[:,None];dN=ne-nm;dZ=ze-zm;etrue=dN-od[:,None]*dZ[...,None]
 # Direct original-token reference in the SAME query-wide scale.
 directw=torch.exp(torch.einsum('qd,ptd->qpt',q,k)/math.sqrt(128)-logscale[:,None,None]);directz=directw.sum(-1);directn=torch.einsum('qpt,ptd->qpd',directw,v);direct_error=max(float((directz-ze).abs().max().item()),float((directn-ne).abs().max().item()))
 cache_error=max(float(np.max(abs(array(ze/total[:,None])-raw['exact_mass']))),float(np.max(abs(array(zm/total[:,None])-raw['summary_mass']))),float(np.max(abs(array(etrue/total[:,None,None])-raw['e']))));assert cache_error<1e-10 and direct_error<1e-8
 methods={'M0':(z0g,n0g),'M1':(z1g,n1g),'M2':(z2g,n2g)};page=[];models={};negative=[];score_store={};errors={}
 maxdelta=delta.abs().amax(-1);std=delta.std(-1,unbiased=False);page_delta=maxdelta.amax(-1);regimes={'all':torch.ones_like(page_delta,dtype=torch.bool),'maxabs_le_0.5':page_delta<=.5,'maxabs_0.5_1':(page_delta>.5)&(page_delta<=1),'maxabs_1_2':(page_delta>1)&(page_delta<=2),'maxabs_gt_2':page_delta>2};risk=torch.as_tensor(raw['risk'],device=device);regimes['risk_unselected']=torch.ones_like(page_delta,dtype=torch.bool)&~risk[None]
 for name,(zg,ng) in methods.items():
  zp=zg.sum(-1);npred=ng.sum(2);pn=npred-nm;pz=zp-zm;ep=pn-ohat[:,None]*pz[...,None];models[name]=(ep,zp);negative.append(dict(method=name,negative_group_mass_fraction=float((zg<=0).double().mean().item()),negative_page_mass_fraction=float((zp<=0).double().mean().item())))
  # Evaluation-only common DenseZ normalization, absent from policy calculations.
  en=ep/total[:,None,None];en_true=etrue/total[:,None,None];pn=pn/total[:,None,None];tn=dN/total[:,None,None];pz=pz/total[:,None];tz=dZ/total[:,None];centering_target=(dN-ohat[:,None]*dZ[...,None])/total[:,None,None]
  for scope,mask in regimes.items():
   if not bool(mask.any()):continue
   def nmse(a,b):return float((a[mask]-b[mask]).square().sum().item()/(b[mask].square().sum().item()+1e-30))
   veca=en[mask];vecb=en_true[mask];norm=veca.norm(dim=-1)*vecb.norm(dim=-1);valid=norm>1e-25;cos=(veca*vecb).sum(-1)[valid]/norm[valid]
   page.append(dict(event_id=eid,split=split(eid),method=name,scope=scope,query_pages=int(mask.sum().item()),deltaZ_nmse=nmse(pz,tz),deltaN_nmse=nmse(pn,tn),net_residual_nmse=nmse(en,en_true),expansion_only_residual_nmse=nmse(en,centering_target),net_direction_cosine=float(cos.mean().item()) if len(cos) else None,predicted_to_true_residual_norm=float(veca.norm().item()/(vecb.norm().item()+1e-30))))
  errors[name]=dict(deltaZ=array(pz),deltaN=array(pn),net_e=array(en))
 models['exact_deltas_online_center']=(dN-ohat[:,None]*dZ[...,None],ze);models['exact_dense_reference']=(etrue,ze)
 panels={'common':(an['remove'],an['add'])};rr=np.flatnonzero(raw['risk']);rr=rr[rr!=int(raw['own'])];aa=np.flatnonzero(~raw['risk']);panels['full_legal']=(np.repeat(rr,len(aa)),np.tile(aa,len(rr)))
 performance=[];selected=[];max_common=0.
 for panel,(r,a) in panels.items():
  truth,_,trueparts=gain_scores(etrue,ze,zm,raw['risk'],r,a,np.arange(64),details=True);truth=truth*64/float(raw['C']);J=float(np.sum(np.square(raw['e'][:,~raw['risk']].sum(1)).sum(-1)/np.where(raw['risk'][None],raw['exact_mass'],raw['summary_mass']).sum(1)**2)/float(raw['C']))
  if panel=='common':max_common=float(np.max(abs(truth-an['gains'][0])));assert max_common<1e-10
  for name,(ep,zp) in models.items():
   for M,qs in [(16,f['probe_indices'].astype(int)),(64,np.arange(64))]:
    score,bad,parts=gain_scores(ep,zp,zm,raw['risk'],r,a,qs,details=True);key=panel+'__'+name+'__Q'+str(M)
    if score is None:performance.append(dict(event_id=eid,split=split(eid),panel=panel,method=name,probes=M,status='INVALID_PREDICTED_DENOMINATOR',invalid_candidates=bad,actions=len(r)));continue
    score_store[key]=score;imse=float(np.sum((parts['I']-trueparts['I'])**2)/(np.sum(trueparts['I']**2)+1e-30));bmse=float(np.sum((parts['B']-trueparts['B'])**2)/(np.sum(trueparts['B']**2)+1e-30));valid=abs(trueparts['I'])>1e-12;isign=float(np.mean((parts['I'][valid]>0)==(trueparts['I'][valid]>0))) if valid.any() else None;met=metrics(score[None],truth[None],np.array([J]));performance.append(dict(event_id=eid,split=split(eid),panel=panel,method=name,probes=M,status='VALID',invalid_candidates=0,actions=len(r),I_nmse=imse,B_nmse=bmse,I_sign_accuracy=isign,**met));j=int(score.argmax());selected.append(dict(event_id=eid,panel=panel,method=name,probes=M,remove=int(r[j]),add=int(a[j]),execute=bool(score[j]>0),pred_score=float(score[j]),true_gain=float(truth[j]),nmse=J,pred_I=float(parts['I'][j]*64/float(raw['C'])),pred_B=float(parts['B'][j]*64/float(raw['C'])),true_I=float(trueparts['I'][j]*64/float(raw['C'])),true_B=float(trueparts['B'][j]*64/float(raw['C']))))
 stats=dict(groups=int(maxdelta.numel()),max_abs_delta_quantiles={str(p):float(torch.quantile(maxdelta.flatten(),p).item()) for p in [0.,.5,.9,.99,1.]},delta_std_quantiles={str(p):float(torch.quantile(std.flatten(),p).item()) for p in [0.,.5,.9,.99,1.]},mean_abs_delta_mean=float(dm.abs().mean().item()),small_group_fraction={str(t):float((maxdelta<=t).double().mean().item()) for t in [.5,1.,2.]},negative_mass=negative)
 np.savez_compressed(out/'scores.npz',**score_store)
 np.savez_compressed(out/'logit_regimes.npz',max_abs_delta=array(maxdelta),std_delta=array(std),mean_delta=array(dm))
 dump(out/'result.json',dict(event_id=eid,split=split(eid),page_metrics=page,performance=performance,selected=selected,logit_statistics=stats,audit=dict(token_ids_unique=True,summary_equal=True,covariance_contraction_error=cov_err,full_second_tensor_error=second_error,raw_direct_contribution_error=direct_error,frozen_contribution_error=cache_error,common_gain_error=max_common,snapshot_sha=sha(path),data_sha=sha(DATA/'contributions'/eid/'residuals_fp64.npz'),grouping='physical reshape(P,4,16,128)',device=device)));print(eid,stats,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--event',required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args();run(args.event,args.device)
