"""Paired mass-underestimation / gradient-allocation reference from E40 cache.
Uses cached Q64 predictions sliced to M16; small replay differences quantified separately.
"""
from .run import D,CACHE,OUT,E29,table,dump
import json,numpy as np,torch
from scipy.special import softmax
from scipy.stats import spearmanr

def run():
 fit=torch.load(OUT/'fit.pt',weights_only=True);floor=fit['bscale'];rows=[];qs=np.rint(np.linspace(0,63,16)).astype(int)
 for seed in [17,29]:
  scale=json.load(open(E29/f's{seed}/local_scale.json'))['scale']
  for split in ['train','development']:
   cd=CACHE/f'{split}_s{seed}';meta=json.load(open(cd/'metadata.json'))
   for mm in meta:
    x=np.load(cd/('event_%04d.npz'%mm['index']));bt=x['beta'][qs];bp=x['predicted_beta'][qs].astype(float);lm=x['logmass'][qs];m=np.exp(lm);true=softmax(lm+bt,axis=1);pred=softmax(lm+bp,axis=1);err=bp-bt;bs=np.maximum(np.sqrt(np.mean(bt**2,axis=0,keepdims=True)),floor)
    gb=np.abs(np.clip(err/bs,-1,1)/bs/err.size);unscaled=np.abs(np.clip(err,-1,1)/err.size);kl=np.abs(pred-true)/16
    lc=m**2*x['local_mse'][qs]/scale**2/err.size
    # Local gradient L2 over the 16 response coordinates, from cached mean squared response error.
    gl=2*m**2*np.sqrt(16*x['local_mse'][qs])/scale**2/(err.size*16)
    base={k:mm[k] for k in ['seed','event','split','trajectory','layer','head','chunk','step','tile']}
    for role in ['all','endpoint_pair','harmful_predicted_endpoint','oracle_endpoint','background']:
     page=x['role_'+role];reg=np.broadcast_to(page,err.shape)
     if not reg.any():continue
     for status,cond in [('all',np.ones_like(reg)),('summary_under10',m<true/10),('predicted_under10',pred<true/10)]:
      sel=reg&cond
      if not sel.any():continue
      row=dict(**base,role=role,status=status,query_pages=int(sel.sum()),sample_fraction=float(sel.mean()),true_mass_share=float(true[sel].sum()/16),summary_mass_share=float(m[sel].sum()/16),predicted_mass_share=float(pred[sel].sum()/16),beta_gradient_share=float(gb[sel].sum()/gb.sum()),local_gradient_l2_share=float(gl[sel].sum()/gl.sum()) if gl.sum() else None,local_loss_share=float(lc[sel].sum()/lc.sum()),kl_gradient_share=float(kl[sel].sum()/kl.sum()),beta_abs_error=float(abs(err[sel]).mean()),beta_scale_mean=float(np.broadcast_to(bs,err.shape)[sel].mean()),huber_linear_fraction=float((abs(err[sel]/np.broadcast_to(bs,err.shape)[sel])>=1).mean()),normalized_to_unscaled_beta_grad=float(gb[sel].sum()/unscaled[sel].sum()) if unscaled[sel].sum() else None)
      rows.append(row)
 table(D/'mass_underestimation_coverage.csv',rows)
 import collections
 gs=collections.defaultdict(list)
 for r in rows:gs[(r['seed'],r['split'],r['role'],r['status'])].append(r)
 out=[]
 for key,rs in gs.items():
  row=dict(zip(['seed','split','role','status'],key));row['events']=len(rs)
  for k in ['sample_fraction','true_mass_share','summary_mass_share','predicted_mass_share','beta_gradient_share','local_gradient_l2_share','local_loss_share','kl_gradient_share','beta_abs_error','beta_scale_mean','huber_linear_fraction','normalized_to_unscaled_beta_grad']:
   z=[r[k] for r in rs if r[k] is not None];row[k+'_mean']=float(np.mean(z));row[k+'_median']=float(np.median(z))
  out.append(row)
 table(D/'mass_coverage_summary.csv',out);print('COVERAGE DONE',len(rows),flush=True)
if __name__=='__main__':run()
