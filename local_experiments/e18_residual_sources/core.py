import json,time,csv,concurrent.futures
from pathlib import Path
import numpy as np
from ..e14_value_observability.core import DATA,E13,load_online,split,sha,dump,cosine
from ..e15_summary_information.run import corr
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/e18_residual_sources_v1';E17=ROOT/'artifacts/e17_swap_decomposition_v1';E15=ROOT/'artifacts/e15_summary_information_v1'
REPS=('full_v','mean_only')
def ids(sp=None):return sorted(p.name for p in (E13/'events').iterdir() if sp is None or split(p.name)==sp)
def table(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def summary_output(f):
 q=f['query'].astype(float);k=f['summary_keys'].astype(float);U=f['basis'].astype(float);mu=f['summary_mean'].astype(float)
 logits=np.einsum('qd,prd->qpr',q,k)/np.sqrt(128);w=np.exp(logits-logits.max((1,2),keepdims=True));mass=w.sum(-1);mass/=mass.sum(1,keepdims=True);within=w/w.sum(-1,keepdims=True)
 haar=np.array([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],float)
 reps=mu[:,None]+np.einsum('ar,pad->prd',haar,f['summary_detail'].astype(float)@U.T)
 op=np.einsum('qpr,prd->qpd',within,reps)
 # Independent contrast-first reconstruction matches capture.py's numerator path.
 coeff=np.einsum('qpr,ar->qpa',within,haar);code=np.einsum('qpa,pak->qpk',coeff,f['summary_detail'].astype(float));op2=mu[None]+np.einsum('qpk,dk->qpd',code,U)
 assert np.allclose(op,op2,rtol=1e-11,atol=1e-12)
 return op,mass

def preregister():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():return
 dump(OUT/'protocol.json',dict(created=time.time(),scope='E18-A/B/C/D, same144 frozen headtiles994 states, no new QKV/attention sampling/video/network/summary/global solve',sources={'data_manifest':sha(DATA/'manifest.json'),'E13_preparation':sha(E13/'preparation.json'),'E17_model':sha(E17/'model_lock.json')},A='ez=d*(summary_page_output-Dense_output), ev=e-ez in DenseZ-normalized coordinates; page energy VV+ZZ+2VZ and actual mixed set energy with true Z,C; all pages, Risk unselected, union of pages in strictly beneficial panel swaps, best beneficial swap pages; labels only for grouping',B='I_VV,I_VZ,I_ZV,I_ZZ at Q64 and matching16; all terms signed, absolute shares separate; report all candidates, true-positive/negative, selected bad/good, panel best; not causal attribution',D='Full-V online proxy only, M16 matched target components; signed/absolute cosine per query before averaging; pages, residual sets, swap directions; retain target total,V,Z; no label features',C='reuse E17 training-fitted lambda without any new ratio or gate search. Convert E17 Iproxy and lambda Bproxy to NMSE-like units by C_hat=4*sum16(norm(all_summary_output)^2), X_I=64*mean16(Iproxy)/C_hat, X_B=64*lambda*mean16(Bproxy)/C_hat. Train ONE nonnegative common multiplier c per representation against both true I and B; preserves original E17 ranking and zero-gate. Weights equal event, equal state-action, per-event divide both errors by mean(I^2+B^2). No evaluation label scale.',C_interventions='base=cX_I-cX_B; teacherI=I-cX_B; teacherB=cX_I-B; teacherIB=I-B; full oracle=I-B+D. All Q64 true terms same NMSE units, no D except full oracle. Repeat teacher replacements with4*true16 terms as query-coverage control; evaluate all on trueQ64 output. No claim of global optimum or deployability.',fit='train city10136 only; development36 diagnostic only no tuning; freeze before horse72 evaluation; original E17 lambda fixed; no scale/threshold grid',report='risk and post-risk separate; equal event then state weights; per-event and pooled statistics; full signed cross terms, oracle interventions conditional on retained proxy, not unconditional information upper bounds'))
