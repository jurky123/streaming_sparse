import json,subprocess
from pathlib import Path
from .common import *
ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest.json').exists(),'Dataset already initialized; do not overwrite'
for n in ('contributions','searches','global_bounds','analyses','logs','trajectories','_validation_spool','representatives'):(ROOT/n).mkdir(exist_ok=True)
es=events();assert len(es)==144 and len({e['event_id'] for e in es})==144
selected=[]
for ci,case in enumerate(('city_walk','horse_field')):
 for si,seed in enumerate((101,303)):
  for li,l in enumerate((1,15,29)):selected.append(event(case,seed,(2,5,8)[(ci+si+li)%3],l,2*((li+ci)%2),(si+li)%2)['event_id'])
weights=['checkpoints/self_forcing_dmd.pt','artifacts/four_gaps_20261031/shared/epoch1.pt','artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'];ident={p:sha(p) for p in weights}
files=['local_experiments/review_v1/ops.py','local_experiments/review_v1/runtime.py','local_experiments/review_v1/model.py','local_experiments/group_response/core.py','local_experiments/group_response/baseline_runtime.py','local_experiments/rollout_suite.py','local_experiments/broad_common.py']
identity=dict(baseline='178f95a80814ff672f59ade8f1c86bf0f037000b',checkpoint='summary_pretrained',grouping='G-KVR',weights=ident,code={p:sha(p) for p in files},inputs={c:sha(f'artifacts/broad_coverage_20261030/embeddings/{c}.pt') for c in ('city_walk','horse_field')},splits={'city_walk':'future fitting/development','horse_field':'mechanism validation, previously seen in development; not unseen test'},head_rule='(layer+chunk+(seed-101)//202+3*step)%12',tile_rule='t0=(17*layer+13*chunk+seed)%72; t1=(t0+36)%72',position_rule='absolute original token id -> latent_frame=id//1536, row=(id%1536)//48, col=id%48')
dump(ROOT/'identity_and_splits.json',identity)
dump(ROOT/'manifest.json',dict(dataset='set_oracle_v1',state='COLLECTING_NOT_FROZEN',events=es,total_events=144,cases=['city_walk','horse_field'],trajectories=4,Q=64,page_tokens=64,R=4,rank=8,history_window=338,current_pages=72,budget='floor(P*0.1), own protected',precision='BF16-deployed QKV/summary -> FP64 reference residual; FP32 deployment separately measured',global_preregistered=selected,global_cpu_cap_hours=12,global_initial_seconds_per_instance=90,global_initial_workers=12,scope='T1+T2 plus data needed for T3; no T3 comparisons or T4 training',contributions_immutable_after_acceptance=True))
print('PREPARED',len(es),'global',len(selected),flush=True)
