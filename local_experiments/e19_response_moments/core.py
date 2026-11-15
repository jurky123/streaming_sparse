import json,time,csv,concurrent.futures
from pathlib import Path
from collections import defaultdict
import numpy as np
from ..e18_residual_sources.core import DATA,E13,load_online,split,sha,dump,summary_output,cosine
from ..e17_swap_decomposition.evaluate import metrics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/e19_response_moments_v1';E17=ROOT/'artifacts/e17_swap_decomposition_v1'
def ids(sp=None):return sorted(p.name for p in (E13/'events').iterdir() if sp is None or split(p.name)==sp)
def components(eid):
 raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));f=load_online(DATA/'contributions'/eid);op,mass=summary_output(f);z=raw['d'][...,None]*(op-raw['dense_output'][:,None]);return raw,f,{'net':raw['e'],'V':raw['e']-z,'Z':z}
def table(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def aggregate(rows,keys):
 by=defaultdict(list)
 for r in rows:by[tuple(r[k] for k in keys)].append(r)
 out=[]
 for vals,rr in sorted(by.items()):
  v=dict(zip(keys,vals));v['rows']=len(rr)
  for k in rr[0]:
   if k in keys or k=='event_id':continue
   a=[r[k] for r in rr if r.get(k) is not None]
   if a and isinstance(a[0],(int,float,np.floating)):v[k]=float(np.mean(a))
  if 'oracle_gain' in v:
   v['pooled_gated_recovery']=v['gated_gain']/v['oracle_gain'] if v['oracle_gain']>1e-30 else None
  out.append(v)
 return out
def preregister():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():return
 dump(OUT/'protocol.json',dict(created=time.time(),scope='E19 A144 frozen headtiles994 sets; B only TWO existing snapshots city_walk/horse_field s101c5l15step0h8t61, no new video/data capture/router',A_fit='Three shared global16-dimensional uncentered PCA/SVD subspaces, on net/V/Z page vectors separately. Train city10136 only. Average per-event second moment divided by its trace; same event weight. No mean subtraction/restoration (linear origin projection preserves residual additivity). Current basis rank8 QR orthonormal span. No rank or covariance-weight selection on dev/test.',A_eval='project all three components into each same subspace; page/set/candidate energy and set direction; decisions use projected NET residual and exact d/Z, Q16 vsQ64, original common legal panels; final actualQ64 NMSE. Oracle representation capacity only, not online prediction.',B_groups='raw physicalPage64.reshape(P,4,16,128), actual learned/recentered K reps and decoded BF16 mean/detail/basis; no regrouping; token IDs and summary caches verified',B_M0='Z0=16 exp(ell)(1+mean(delta)); N0=16 exp(ell)(1+mean(delta))*meanV; linear K mean offset plus exact V mean offset, no covariance',B_M1='same Z asM0; N1=16 exp(ell)*mean((1+delta)*V) = N0 +16 exp(ell)Cov(V,K)q/sqrt128',B_M2='Z2=16 exp(ell)*mean(1+delta+delta^2/2); N2=16 exp(ell)*mean((1+delta+delta^2/2)*V), includes full mixed delta^2V, not just Key variance',B_scale='shared per-query online summary total logZ for ALL group/page N/Z; delta estimates subtract original four representatives. ehat=deltaNhat-Ohat_all deltaZhat; predicted mixed denominator sums estimated exact Z on selected pages and original summary Z elsewhere. No Dense output/true d in moment policy.',B_exact_controls='exact deltas centered by online Ohat_all (centering-only error diagnostic), and exact Dense-centered residual/full true denominator oracle',B_metrics='delta distribution and regimes maxabsdelta<=.5,<=1,<=2,>2; DeltaN/DeltaZ relative squared error; residual cosine/norm error; initial Risk common panel and full legal one-swap panel, Q16/64 scores evaluated trueQ64. No-op if top score<=0. Invalid predicted denominators NOT clipped: whole method/panel marked invalid if any baseline/candidate denominator<=0.',B_capacity='full moment contractions computed offline from16 raw tokens, first covariance explicitly formed and audited; second delta^2V contraction includes full tensor without materializing it. Not proposed cache size or deployable latency.',frozen_manifest_sha=sha(DATA/'manifest.json')))
