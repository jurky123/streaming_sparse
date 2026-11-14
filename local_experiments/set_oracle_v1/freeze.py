"""One-time acceptance of immutable contributions, independent checks of saved search labels."""
import concurrent.futures,json,time,os,datetime
from pathlib import Path
import numpy as np
from .common import *

def check_event(eid):
 root=ROOT/'contributions'/eid;ident=json.loads((root/'identity.json').read_text());cap=json.loads((root/'capture_validation.json').read_text());res=dict(np.load(root/'residuals_fp64.npz'));online=dict(np.load(root/'summary_and_route_features.npz'));r=ROOT/'searches'/'v1'/eid;s=json.loads((r/'result.json').read_text());v=json.loads((r/'validation.json').read_text());assert cap['passed'] and v['passed'];P=ident['P'];K=ident['K'];own=int(res['own']);e=res['e'];d=res['d'];C=float(np.square(res['dense_output']).sum()+1e-12)
 assert e.shape==(64,P,128) and e.dtype==np.float64 and d.shape==(64,P);assert K==int(P*.1)
 for name,a in res.items():assert np.isfinite(a).all(),(eid,name)
 for name,a in online.items():assert np.isfinite(a).all(),(eid,name)
 assert np.all(res['exact_mass']>0) and np.all(res['summary_mass']>0);assert np.max(np.abs(res['exact_mass'].sum(1)-1))<1e-12;assert np.allclose(d,res['exact_mass']-res['summary_mass'],atol=1e-12,rtol=1e-12)
 assert online['risk_features_q64'].shape==(64,P,4) and online['summary_keys'].shape==(P,4,128) and online['summary_detail'].shape==(P,3,8)
 for name in ('query','summary_keys','summary_mean','summary_detail','summary_stats','basis'):assert np.all((online[name].view(np.uint32)&65535)==0),(eid,name,'not exact BF16 values')
 ids=online['token_ids'];qids=online['query_token_ids'];assert ids.shape==(P,64) and len(np.unique(ids))==P*64 and np.array_equal(ids[own],qids);assert ident['start_token']==ident['chunk']*4608;assert np.array_equal(online['token_latent_frame'],ids//1536) and np.array_equal(online['token_y'],ids%1536//48) and np.array_equal(online['token_x'],ids%48)
 for f,h in ident['files'].items():assert sha(root/f)==h,(eid,f,'hash changed')
 for method,j in s['values'].items():
  x=np.zeros(P,bool);x[s['masks'][method]]=True;assert x.sum()==K and x[own];E=e[:,~x].sum(1);z=np.where(x[None],res['exact_mass'],res['summary_mass']).sum(1);actual=float(np.sum(np.square(E).sum(1)/z**2)/C);assert abs(actual-j)<max(1e-11,abs(j)*1e-8)
 surfaces=dict(np.load(r/'candidate_gain_surfaces.npz'));g=surfaces['gains'];masks=surfaces['state_masks'];rr=surfaces['removed'];aa=surfaces['added'];assert g.shape==(len(s['trajectories']['risk']),K-1,P-K);assert masks.shape==(len(g),P);max_surface_error=0.
 rng=np.random.default_rng(17)
 for i,x in enumerate(masks):
  assert x.sum()==K and x[own] and np.array_equal(rr[i],np.where(x&(np.arange(P)!=own))[0]) and np.array_equal(aa[i],np.where(~x)[0]);assert hashlib.sha256(x.tobytes()).hexdigest()==s['top_candidates']['risk'][i]['state_sha256'];E=e[:,~x].sum(1);z=np.where(x[None],res['exact_mass'],res['summary_mass']).sum(1);j=float(np.sum(np.square(E).sum(1)/z**2)/C)
  for ri,ai in zip(rng.integers(0,K-1,3),rng.integers(0,P-K,3)):
   remove,add=rr[i,ri],aa[i,ai];EE=E+e[:,remove]-e[:,add];zz=z-d[:,remove]+d[:,add];gain=j-float(np.sum(np.square(EE).sum(1)/zz**2)/C);err=abs(gain-g[i,ri,ai]);max_surface_error=max(max_surface_error,err);assert err<max(1e-11,abs(j)*1e-8)
 d02=dict(np.load(r/'D02.npz'));assert np.max(np.abs(d02['weighted_components'].sum((1,2))-d02['nmse']))<1e-10
 return dict(event_id=eid,P=P,K=K,head=ident['head'],reused=ident['reused_contributions'],contribution_files=ident['files'],identity_sha256=sha(root/'identity.json'),capture_validation_sha256=sha(root/'capture_validation.json'),max_direct_error=max(v['max_error'],max(t['absolute_difference'] for t in cap['tests'])),surface_sample_max_error=max_surface_error,fp32=cap['fp32'],risk_steps=len(s['trajectories']['risk'])-1,all_finite=True,identity_passed=True)
if __name__=='__main__':
 begin=time.monotonic();manifest=json.loads((ROOT/'manifest.json').read_text());assert manifest['state']=='COLLECTING_NOT_FROZEN';expected=[e['event_id'] for e in manifest['events']];assert len(list((ROOT/'contributions').glob('*/READY.json')))==len(expected)==144
 for case in ('city_walk','horse_field'):
  for seed in (101,303):assert json.loads((ROOT/'trajectories'/f'{case}_s{seed}'/'status.json').read_text())['status']=='complete'
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(check_event,expected))
 assert not list((ROOT/'_validation_spool').glob('*.npz')),'unconsumed original contributions'
 sources={str(p):sha(p) for p in Path('local_experiments/set_oracle_v1').glob('*.py')};dump(ROOT/'acceptance.json',dict(passed=True,events=rows,event_count=144,max_direct_error=max(r['max_direct_error'] for r in rows),max_surface_error=max(r['surface_sample_max_error'] for r in rows),reused_events=sum(r['reused'] for r in rows),head_coverage=sorted({r['head'] for r in rows}),wall_seconds=time.monotonic()-begin,collector_and_analysis_code=sources))
 manifest.update(state='FROZEN_VALIDATED',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),acceptance_sha256=sha(ROOT/'acceptance.json'),identity_sha256=sha(ROOT/'identity_and_splits.json'),contribution_index=[dict(event_id=r['event_id'],files=r['contribution_files'],identity_sha256=r['identity_sha256']) for r in rows]);dump(ROOT/'manifest.json',manifest)
 for f in (ROOT/'contributions').rglob('*'):
  if f.is_file():os.chmod(f,0o444)
 print('FROZEN',len(rows),'max error',max(r['max_direct_error'] for r in rows),'seconds',time.monotonic()-begin,flush=True)
