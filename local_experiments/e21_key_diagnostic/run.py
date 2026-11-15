from ..e20_compressed_response.core import *
from ..e20_compressed_response import core as prev
from ..e12_swap_prediction.core import objective
import argparse
OUT=ROOT/'artifacts/e21_key_diagnostic_v1'

def setup():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():return
 dump(OUT/'protocol.json',dict(scope='Two existing E20 snapshots, full V, original M2 locked by E20. Q64/Page64/P410/K41/own. No videos, training networks, new cache or inference code.',basis_rules={'bankQR8':'E20 fixed QR first8','bankSVD8':'left singular vectors of full 128x32 frozen bank, descending singular values','trainQ8':'uncentered second moment PCA of ALL64 queries in the 36 city_walk seed101 TRAIN events, each event equal weight; shared across heads. Neither city303 nor horse used.','snapshotKOracle8':'per-snapshot PCA of all raw k-kappa vectors, train and evaluate on same snapshot; capacity diagnostic ONLY'},rank16='Only if no non-oracle rank8 passes M16>=80% fullM2 positive in BOTH snapshots: one trainQ16 (same training eigensystem), no other rank16 search.',quality='M16 primary, Q64 control. Passing K-only/fullV is not low-cost sketch validation. No post-hoc chosen basis generalized from two cases.',B='Full legal and HH/HC/CH/CC panels; true raw gains, full-M2 scores versus missing-current response. H/C arrow means remove->add. Restricted HH distinguishes action coverage from missing-current scoring. Additional HH controls zero only current residual E or only current denominator correction, not trainable methods.',precision='FP32 bases, FP64 direct polynomial contractions; no BF16 cache implementation. Costs hypothetical BF16 fullV and V8 for ranks8/16.',created=time.time()))
 state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model'];bank=state['layers.15.bank'][8].double().numpy().T;u,sv,_=np.linalg.svd(bank,full_matrices=False);cov=np.zeros((128,128));sources={}
 ids=sorted(p.name for p in (DATA/'contributions').glob('city_walk_s101_*'));assert len(ids)==36
 for eid in ids:
  p=DATA/'contributions'/eid/'summary_and_route_features.npz';q=np.load(p)['query'].astype(float);assert q.shape==(64,128);cov+=q.T@q/(64*len(ids));sources[eid]=sha(p)
 eig,U=np.linalg.eigh(cov);U=U[:,::-1];old=np.load(prev.OUT/'bases.npz');np.savez(OUT/'bases.npz',bankQR8=old['key8'],bankSVD8=u[:,:8].astype('float32'),trainQ8=U[:,:8].astype('float32'),trainQ16=U[:,:16].astype('float32'),trainQ_spectrum=eig[::-1],bank_spectrum=sv)
 dump(OUT/'basis_lock.json',dict(train_ids=ids,train_sources=sources,bases_sha=sha(OUT/'bases.npz'),checkpoint_sha=sha(CHECKPOINT),protocol_sha=sha(OUT/'protocol.json')))

def run(eid,device,rank16=False):
 ctx=load(eid,device);dest=OUT/eid;dest.mkdir(exist_ok=True);T=ctx['T'];base=np.load(OUT/'bases.npz');delta=torch.einsum('qd,pgtd->qpgt',ctx['q'],ctx['dk'])/math.sqrt(128);methods=[];scores={};response=[];audits=[]
 configs={'trainQ16':base['trainQ16']} if rank16 else {n:base[n] for n in ['bankQR8','bankSVD8','trainQ8']}
 if not rank16:
  dk=array(ctx['dk']).reshape(-1,128);_,vec=np.linalg.eigh(dk.T@dk/len(dk));configs['snapshotKOracle8']=vec[:,-8:][:,::-1].astype('float32');np.save(dest/'snapshotKOracle8.npy',configs['snapshotKOracle8'])
 for name,U in configs.items():
  uk=T(U);pred=torch.einsum('qk,pgtk->qpgt',ctx['q']@uk,ctx['dk']@uk)/math.sqrt(128);err=pred-delta
  response.append(dict(event_id=eid,method=name,delta_nmse=float(err.square().sum()/delta.square().sum()),delta_mae=float(err.abs().mean()),delta_p99_abs_error=float(torch.quantile(err.abs().flatten(),.99)),delta_max_abs_error=float(err.abs().max()),oracle=name.startswith('snapshot')))
  e,z,meta=direct(ctx,'original',uk=uk);r=evaluate(ctx,name,e,z,meta=meta);scores.update(r.pop('scores'));methods.append(r)
  for a in r['selected']:
   mask=ctx['raw']['risk'].copy()
   if a['execute']:mask[a['remove']]=False;mask[a['add']]=True
   g=ctx['J']-objective(ctx['raw'],mask);assert abs(g-(a['true_gain'] if a['execute'] else 0))<1e-10;assert mask.sum()==41 and mask[int(ctx['raw']['own'])];audits.append(abs(g-(a['true_gain'] if a['execute'] else 0)))
  if name=='bankQR8':
   old=np.load(prev.OUT/'compression'/eid/'scores.npz')
   for pan in ctx['panels']:
    for M in [16,64]:assert np.max(abs(scores[name+'__all_pages__'+pan+'__Q'+str(M)]-old['K_only8__all_pages__'+pan+'__Q'+str(M)]))<1e-10
 tag='rank16' if rank16 else 'rank8';np.savez_compressed(dest/(tag+'_scores.npz'),**scores);dump(dest/(tag+'.json'),dict(event_id=eid,methods=methods,response=response,selected_audit_max=max(audits),bases_sha=sha(OUT/'bases.npz')))
 if rank16:return
 e,z,_=direct(ctx,'original');h=torch.as_tensor(ctx['f']['page_is_history'],device=device);ee=e.clone();ee[:,~h]=0;zz=z.clone();zz[:,~h]=ctx['zh'][:,~h];r,a=ctx['panels']['full_legal'];hist=ctx['f']['page_is_history'];groups=np.array([('H' if hist[x] else 'C')+('H' if hist[y] else 'C') for x,y in zip(r,a)]);truth=ctx['truth']['full_legal'];b=[]
 variants={'full_M2':(e,z),'history_only':(ee,zz),'missing_current_E_only':(ee,z),'missing_current_Z_only':(e,zz)}
 for name,(ev,zv) in variants.items():
  for M,qs in [(16,ctx['f']['probe_indices'].astype(int)),(64,np.arange(64))]:
   sc,bad=gain_scores(ev,zv,ctx['zh'],ctx['raw']['risk'],r,a,qs);assert sc is not None and bad==0
   for group in ['ALL','HH','HC','CH','CC']:
    ix=np.arange(len(r)) if group=='ALL' else np.flatnonzero(groups==group)
    if not len(ix):continue
    j=ix[sc[ix].argmax()];oj=ix[truth[ix].argmax()];gain=float(truth[j]) if sc[j]>0 else 0;oracle=max(0,float(truth[oj]));b.append(dict(event_id=eid,method=name,probes=M,group=group,count=len(ix),positive_true_actions=int((truth[ix]>0).sum()),oracle_gain=oracle,oracle_relative_pct=100*oracle/ctx['J'],actual_gain=gain,actual_relative_pct=100*gain/ctx['J'],regret=oracle-gain,remove=int(r[j]),add=int(a[j]),selected_group=groups[j],oracle_remove=int(r[oj]),oracle_add=int(a[oj]),oracle_group=groups[oj]))
 dump(dest/'lifecycle.json',b);print(eid,tag,'done',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--event');p.add_argument('--device',default='cuda:0');p.add_argument('--rank16',action='store_true');a=p.parse_args()
 if a.setup:setup()
 else:run(a.event,a.device,a.rank16)
