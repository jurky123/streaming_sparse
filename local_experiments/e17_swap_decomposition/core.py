import json,time,csv,concurrent.futures
from pathlib import Path
import numpy as np
from ..e15_summary_information.run import proxies,corr
from ..e14_value_observability.core import DATA,E13,load_online,split,sha,dump
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/e17_swap_decomposition_v1';E15=ROOT/'artifacts/e15_summary_information_v1'
REPS=('full_v','mean_only')
def table(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def ids(sp=None):return sorted(p.name for p in (E13/'events').iterdir() if sp is None or split(p.name)==sp)
def preregister():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():return
 dump(OUT/'protocol.json',dict(created=time.time(),name='E17 (user-proposed E16; E16 true-mass already exists)',scope='same144 headtiles994 sets, common legal panels; no new states/QKV/video/network/global solver',A='exact gain = I - B + D; I=-2E.dot(de)/Z^2/C, B=norm(de)^2/Z^2/C, D=norm(E+de)^2*(1/Z^2-1/Znew^2)/C; normalize all by DenseZ as cached; store candidate terms Q64 and matching Q16 sums; validate against original gains',B='full_v and mean_only; score Iproxy-lambda Bproxy, Iproxy=mean16(2H.v), Bproxy=mean16(norm(v)^2); original lambda1 and one train-fitted lambda per representation; no rank8 retraining',fit='single global nonnegative lambda, shared all layers/states; profile nonnegative least squares a*I-b*B to real Q64 gain, lambda=b/a; positive common a is nuisance and never used online. Per-event labels/stdG and features/common online RMS; equal event and equal state-action weights. Analytic 2-variable NNLS, no grid. If a=0,b>0 classify infinite lambda limit (no positive action); if both zero zero-information no-op',selection='development initial Risk36 only, choose lambda1 or train-fitted by mean true relative gain with fixed score>0 gate; tie favors baseline; freeze before evaluation; report both even if negative; no additional threshold',split='city101 train36, city303 development36, horse101/303 evaluation72',denominator_diagnostic='lambda1 unchanged direction/amplitude; true current Z weights with deltaZ=0 and full true Z,Znew panels. mean16(||H||^2/Z^2-||-H+v||^2/Znew^2). Offline teacher information, not deployable; no mixing raw FP32 outputs with FP64 goal',metrics='risk separately from post-risk and all; equal event then state weight; candidate sign accuracy excluding numerical ties, selected-action actual gain/regret/recovery/degradation, positive-label filtering diagnostic; distributions of exact I,B,D; no causality claim from decomposition',sources={'manifest':sha(DATA/'manifest.json'),'E13':sha(E13/'preparation.json'),'E15':sha(E15/'protocol.json')}))
def score(d,rep,lam):
 I=d[rep+'_I'];B=d[rep+'_B']
 if lam is None:return -B
 return I-lam*B
