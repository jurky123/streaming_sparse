from .core import *
from .fit_pca import one as fit_one
from ..e12_swap_prediction.core import swap_labels,objective
import torch

def capacity_one(eid):
 result=json.loads((OUT/'capacity'/eid/'result.json').read_text());assert result['audit']['pca_sha']==sha(OUT/'pca16.npz')
 for p,h in result['audit']['sources'].items():assert sha(ROOT/p)==h
 scores=np.load(OUT/'capacity'/eid/'scores.npz');st=np.load(E13/'events'/eid/'states.npz');an=np.load(E13/'events'/eid/'anchors.npz');raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));err=0
 for i,x in enumerate(st['masks']):
  js=np.unique([int(scores[n][i].argmax()) for n in scores.files]);labels=swap_labels(raw,x,an['remove'][js],an['add'][js]);err=max(err,float(np.max(abs(labels-an['gains'][i,js]))))
 assert err<1e-10
 return err

def run():
 lock=json.loads((OUT/'pca_lock.json').read_text());assert lock['train_ids']==ids('train') and lock['sha256']==sha(OUT/'pca16.npz');pca=np.load(OUT/'pca16.npz');ms={n:np.zeros((128,128)) for n in ['net','V','Z']}
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
  for eid,mat in ex.map(fit_one,ids('train')):
   for n in ms:ms[n]+=mat[n]/36
 eigerr=0.
 for n,m in ms.items():
  U=pca[n];eigs=pca[n+'_spectrum'][:16];assert np.allclose(U.T@U,np.eye(16),atol=1e-12);err=float(np.max(abs(m@U-U*eigs)));assert err<1e-10;eigerr=max(eigerr,err)
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:gainerr=max(ex.map(capacity_one,ids()))
 snapshots=[]
 for path in sorted((OUT/'moments').glob('*/result.json')):
  result=json.loads(path.read_text());eid=result['event_id'];snap=torch.load(DATA/'representatives'/f'{eid}.pt',weights_only=True,map_location='cpu');f=dict(np.load(DATA/'contributions'/eid/'summary_and_route_features.npz'));raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));assert sha(DATA/'representatives'/f'{eid}.pt')==result['audit']['snapshot_sha'];assert sha(DATA/'contributions'/eid/'residuals_fp64.npz')==result['audit']['data_sha']
  kg=snap['k'].float().reshape(-1,4,16,128);vg=snap['v'].float().reshape_as(kg);kap=torch.from_numpy(f['summary_keys']);mu=torch.from_numpy(f['summary_mean']);detail=torch.from_numpy(f['summary_detail']);basis=torch.from_numpy(f['basis']);haar=torch.tensor([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=torch.float32);nu=mu[:,None]+torch.einsum('ag,pak,dk->pgd',haar,detail,basis);bk=(kg.mean(2)-kap).norm(dim=-1);bv=(vg.mean(2)-nu).norm(dim=-1);stats=torch.from_numpy(f['summary_stats']);relative=[]
  for field,calc in [(8,bk),(9,bv)]:
   rel=float(((calc-stats[...,field]).abs()/(calc.abs()+1e-6)).max());assert rel<.006;relative.append(rel)
  error=0
  for row in result['selected']:
   x=raw['risk'].copy();x[row['remove']]=False;x[row['add']]=True;j=objective(raw,raw['risk'])-objective(raw,x);error=max(error,abs(j-row['true_gain']))
  assert error<1e-10
  historical_check=path.with_name('result_before_IB.json').exists()
  if historical_check:
   old=json.loads(path.with_name('result_before_IB.json').read_text());assert len(old['selected'])==len(result['selected'])
   for x,y in zip(old['selected'],result['selected']):assert (x['remove'],x['add'],x['execute'])==(y['remove'],y['add'],y['execute'])
  snapshots.append(dict(event_id=eid,selected_true_gain_error=error,group_mean_stat_max_relative_errors=relative,IB_diagnostic_did_not_change_decisions=True if historical_check else None,**result['audit']))
 original=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for n,h in ex.map(lambda n:(n,sha(DATA/n)),original):assert h==original[n]
 n=0
 for e in json.loads((E13/'preparation.json').read_text())['events']:
  for name,h in e['files'].items():assert sha(E13/'events'/e['event_id']/name)==h;n+=1
 dump(OUT/'audit.json',dict(passed=True,A_events=144,A_states=994,B_snapshots=len(snapshots),frozen_source_files=len(original),frozen_E13_files=n,train_only_PCA_verified=True,max_covariance_eigen_residual=eigerr,max_A_selected_gain_error=gainerr,max_A_full128_gain_error=json.loads((OUT/'A_complete.json').read_text())['max_full128_error'],snapshot_checks=snapshots));print('AUDIT PASSED',snapshots)
if __name__=='__main__':run()
