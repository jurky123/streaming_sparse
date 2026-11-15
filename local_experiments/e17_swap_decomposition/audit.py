from .core import *
from ..e12_swap_prediction.core import swap_labels,ALLOW
import tempfile

def one(eid):
 ready=json.loads((OUT/'events'/eid/'READY.json').read_text());assert ready['terms_sha']==sha(OUT/'events'/eid/'terms.npz')
 for p,h in ready['sources'].items():assert sha(ROOT/p)==h
 d=np.load(OUT/'events'/eid/'terms.npz');st=np.load(E13/'events'/eid/'states.npz');raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));lock=json.loads((OUT/'model_lock.json').read_text());allscore=[d['gains']]
 for m in lock['models']:
  rep=m['representation'];allscore.extend([score(d,rep,1),score(d,rep,m['lambda_fitted']),d[rep+'_current_z'],d[rep+'_full_z']])
 maxerr=0
 for i,x in enumerate(st['masks']):
  js=np.unique([int(s[i].argmax()) for s in allscore]+[0,d['gains'].shape[1]//2,d['gains'].shape[1]-1]);v=swap_labels(raw,x,d['remove'][js],d['add'][js]);maxerr=max(maxerr,float(np.max(abs(v-d['gains'][i,js]))))
 assert maxerr<1e-10
 return dict(event_id=eid,states=len(st['masks']),candidate_gains=int(d['gains'].size),max_direct_error=maxerr,max_three_term_error=float(np.max(abs(d['I']-d['B']+d['D']-d['gains']))))
def run():
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:rows=list(ex.map(one,ids()))
 lock=json.loads((OUT/'model_lock.json').read_text());assert lock['train_ids']==ids('train') and lock['development_ids']==ids('validation')
 for m in lock['models']:
  A=np.zeros((2,2));b=np.zeros(2)
  for eid in lock['train_ids']:
   d=np.load(OUT/'events'/eid/'terms.npz');ip=d[m['representation']+'_I'];bp=d[m['representation']+'_B'];sc=max(float(np.sqrt(np.mean(ip**2+bp**2))),1e-15);X=np.stack((ip,-bp),-1).reshape(-1,2)/sc;y=d['gains'].ravel()/max(float(d['gains'].std()),1e-12);A+=X.T@X/len(y)/36;b+=X.T@y/len(y)/36
  assert np.allclose(A,m['normal_matrix']) and np.allclose(b,m['normal_rhs']);coef=np.array(m['coefficient']);grad=A@coef-b;assert (coef>=0).all() and (grad[coef==0]>=-1e-8).all() and np.linalg.norm(grad[coef>0])<1e-8
  assert abs(coef[1]/coef[0]-m['lambda_fitted'])<1e-12
 f=load_online(DATA/'contributions'/ids()[0]);original_fields=dict(np.load(DATA/'contributions'/ids()[0]/'summary_and_route_features.npz'))
 for k in original_fields:
  if k not in ALLOW:original_fields[k]=np.zeros_like(original_fields[k])
 with tempfile.TemporaryDirectory() as td:
  np.savez(Path(td)/'summary_and_route_features.npz',**original_fields);ff=load_online(Path(td))
 a,_=proxies(f);b,_=proxies(ff)
 for rep in REPS:assert np.array_equal(a[rep],b[rep])
 original=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for name,h in ex.map(lambda n:(n,sha(DATA/n)),original):assert original[name]==h
 count=0
 for e in json.loads((E13/'preparation.json').read_text())['events']:
  for n,h in e['files'].items():assert sha(E13/'events'/e['event_id']/n)==h;count+=1
 s=json.loads((OUT/'summary.json').read_text());assert s['model_lock_sha256']==sha(OUT/'model_lock.json')
 result=dict(passed=True,events=len(rows),states=sum(r['states'] for r in rows),candidate_gains=sum(r['candidate_gains'] for r in rows),max_direct_error=max(r['max_direct_error'] for r in rows),max_three_term_error=max(r['max_three_term_error'] for r in rows),frozen_base_files=len(original),frozen_E13_files=count,teacher_poisoning_pass=True,train_only_fit_KKT_pass=True,model_frozen_sha=sha(OUT/'model_lock.json'),per_event=rows)
 dump(OUT/'audit.json',result);print({k:v for k,v in result.items() if k!='per_event'})
if __name__=='__main__':run()
