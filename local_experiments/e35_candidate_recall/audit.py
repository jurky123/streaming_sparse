from .run import *
import heapq

def main():
 rows=list(csv.DictReader((DEST/'topL_per_event.csv').open()));pools=list(csv.DictReader((DEST/'restricted_pool_per_event.csv').open()));sources=json.loads((DEST/'sources.json').read_text());lookup={(int(r['seed']),r['event'],r['policy'],int(r['L'])):r for r in rows};plook={(int(r['seed']),r['event'],r['pool']):r for r in pools};checks=[];longchecks=[]
 for seed in [17,29]:
  for idx in range(288):
   path=DEST/f's{seed}/event_{idx:04d}.npz';x=np.load(path);event=str(x['event']);g=x['true_gain'];r=x['remove'];a=x['add'];m=x['mask'];oracle=max(0.,max(float(t) for t in g));orders={}
   src=next(s for s in sources if s['seed']==seed and s['event']==event);assert src['panel_sha']==sha(SRC/f's{seed}/panel_{idx:04d}.npz') and src['beta_sha']==sha(SRC/f's{seed}/beta_{idx:04d}.pt')
   for policy,key in [('Set','set_score'),('Independent','independent_score')]:
    # Independent stable heap ranking rather than NumPy argsort.
    oo=heapq.nlargest(8,range(512),key=lambda j:(float(x[key][j]),-j));orders[policy]=oo;assert np.array_equal(oo,x['set_order' if policy=='Set' else 'independent_order'][:8]);prev=0.
    for L in LS:
     z=lookup[seed,event,policy,L];ii=oo[:L];gain=max(0.,max(float(g[j]) for j in ii));assert abs(gain-float(z['best_gain']))<1e-12 and gain>=prev and gain<=oracle+1e-12;assert int(z['unique_pages'])==len({int(p) for j in ii for p in [r[j],a[j]]});prev=gain
     assert bool(z['has_positive']=='True')==any(g[j]>1e-12 for j in ii)
   union=sorted(set(orders['Set'][:2]+orders['Independent'][:2]));rs={int(r[j]) for j in union};aa={int(a[j]) for j in union};cart=[j for j in range(512) if int(r[j]) in rs and int(a[j]) in aa];assert np.array_equal(union,x['proposal_union']) and np.array_equal(cart,x['touched_page_pool']);assert len(rs)+len(aa)<=8 and len(cart)==len(rs)*len(aa)<=16;assert all(m[j] for j in rs) and all(not m[j] for j in aa)
   for name,ii in [('Four-proposal-union',union),('Touched-page-cartesian',cart)]:
    gain=max(0.,max(float(g[j]) for j in ii));z=plook[seed,event,name];assert abs(gain-float(z['best_gain']))<1e-12
   checks.append(dict(seed=seed,event=event,topL_nested=True,oracle_monotonic=True,no_true_labels_in_order=True,union_and_cartesian_separate=True,source_hashes_equal=True))
   if idx not in [0,47,143,144,230,287]:continue
   cache=torch.load(SRC/f's{seed}/beta_{idx:04d}.pt',map_location='cpu',weights_only=True);qs=np.linspace(0,63,16).round().astype(int)
   def arr(t):return t[qs].double().numpy().astype(np.longdouble)
   lm=arr(cache['logmass']);beta=arr(cache['betas']['Predicted']);response=arr(cache['predicted_response']);center=arr(cache['center']);le=lm+beta;mx=le.max(1,keepdims=True);logd=mx+np.log(np.exp(le-mx).sum(1,keepdims=True));target=(np.exp(le-logd)[...,None]*(center+response)).sum(1);extra=max(0,(lm-logd).max());net=np.exp(le-logd-extra)[...,None]*(center+response-target[:,None])-np.exp(lm-logd-extra)[...,None]*(center-target[:,None]);en=(net*net).sum(-1).mean(0);scores=en[a]-en[r];err=float(np.max(np.abs(scores-x['independent_score'])));scale=max(1,float(np.max(np.abs(scores))));assert err/scale<1e-12;oo=heapq.nlargest(2,range(512),key=lambda j:(scores[j],-j));same=oo==orders['Independent'][:2]
   if not same:assert max(float(abs(scores[j]-scores[k])) for j,k in zip(oo,orders['Independent'][:2]))<1e-12*scale
   longchecks.append(dict(seed=seed,event=event,independent_score_scaled_error=err/scale,top2_same=same))
 done=json.loads((DEST/'complete.json').read_text());assert done['protocol_sha']==sha(DEST/'protocol.json');dump(DEST/'independent_audit.json',dict(events=len(checks),topL_rows=len(rows),restricted_pool_rows=len(pools),checks=checks,longdouble_checks=longchecks,raw_QKV_loaded=False,attention_executed=False,test_accessed=False));print('AUDIT PASSED',len(checks),len(rows),len(pools))
if __name__=='__main__':main()
