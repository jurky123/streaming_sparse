import concurrent.futures,tempfile
from .run import *
from ..e12_swap_prediction.core import ALLOW

def run():
 rows=[json.loads(p.read_text()) for p in sorted((OUT/'events').glob('*/result.json'))];assert len(rows)==144
 for row in rows:
  for name,h in row['audit']['source_hashes'].items():assert sha(ROOT/name)==h
  for state in row['A']:
   assert abs(state['conditional_gain']-state['fixed_strong']-state['V_strong'])<1e-12
   assert state['V_strong']>=-1e-12
  for b in row['B']:assert b['centering_residual']<1e-10
 f0=load_online(DATA/'contributions'/rows[0]['event_id']);full=dict(np.load(DATA/'contributions'/rows[0]['event_id']/'summary_and_route_features.npz'))
 for k in full:
  if k not in ALLOW:full[k]=np.zeros_like(full[k])
 with tempfile.TemporaryDirectory() as td:
  np.savez(Path(td)/'summary_and_route_features.npz',**full);f1=load_online(Path(td))
 p0,u=proxies(f0);p1,_=proxies(f1)
 for k in p0:assert np.array_equal(p0[k],p1[k])
 original=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for name,h in ex.map(lambda name:(name,sha(DATA/name)),original):assert h==original[name]
 count=0
 for event in json.loads((E13/'preparation.json').read_text())['events']:
  for name,h in event['files'].items():assert sha(E13/'events'/event['event_id']/name)==h;count+=1
 # E14 rank8 proxy score differences equal -2 times its registered online dot differences.
 maxerr=0
 for row in rows:
  eid=row['event_id'];old=np.load(E14/'events'/eid/'probes.npz');new=np.load(OUT/'events'/eid/'scores.npz')['rank8'];a=new-new[:1];b=-2*(old['online_dot']-old['online_dot'][:1]);err=float(np.max(abs(a-b)));assert np.allclose(a,b,rtol=1e-9,atol=1e-10);maxerr=max(maxerr,err)
 result=dict(passed=True,events=len(rows),states=sum(len(r['A']) for r in rows),max_direct_gain_error=max(r['audit']['max_gain_error'] for r in rows),max_E14_score_difference_error=maxerr,teacher_poisoning_pass=True,source_base_hashes=len(original),source_E13_hashes=count,no_fitted_parameters=True)
 dump(OUT/'audit.json',result);print(result)
if __name__=='__main__':run()
