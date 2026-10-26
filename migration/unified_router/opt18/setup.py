import json
from pathlib import Path
from ..common import atomic_json,sha
ROOT=Path('streaming_sparse/v10/experiments/opt18_continue_frontier_20261025').resolve()
PARENT=Path('streaming_sparse/v10/experiments/opt17_grouped_joint_20261025').resolve()

def main():
    if (ROOT/'experiment.json').exists():print(ROOT);return
    p=json.loads((PARENT/'experiment.json').read_text())
    m=dict(p,parent_opt17=str(PARENT),max_updates=2000,min_updates=1000,learning_rate_router=1e-5,warmup_updates=25,
           validate_every=100,checkpoint_every=25,tree_refresh_every=0,
           initialization='each Opt17 best model including trees; fresh AdamW because best does not contain optimizer state; 2000 additional steps',
           frozen=p['frozen']+['best Q/K grouping trees during continuation'],alternating=[],
           counterfactual=None,feature_cache=None,
           training_objective='same actual deployed hard attention NMSE as Opt17; gradient joint prototype/projection optimization with best trees fixed',
           sweep_fractions=[.25,.35,.5,.75,1.],sweep_representatives=[4,8,16],
           sweep_new_depths=[6,7,8],sweep_new_tree_fractions=[.35,.5,.75],
           sweep_accuracy_events='all96 Val; full4680-query grouping; original128 sampled loss queries',
           sweep_timing_events='two Val prompts x three history lengths at step3; paired CUDA graph timings, include current rebuild and cold KV rebuild separately',
           sweep_tree_calibration='Train only, same12 events and current learned projection; fresh64 control plus128/256 buckets, no gradient fine-tuning of new trees',
           parallelism='8 independent continuation workers then8 independent frontier workers, one per layer/variant',
           test_used=False)
    m.pop('aggressive',None)
    ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'experiment.json',m)
    locked=json.loads((PARENT/'source_manifest.json').read_text());locked.update(json.loads((PARENT/'parent_lock.json').read_text()))
    sources={}
    for layer in m['layers']:
        for variant in m['variants']:
            f=PARENT/f'training/layer{layer:02d}/{variant}/best.pt';record=json.loads((f.parent/'best_validation.json').read_text())
            locked[str(f)]=sha(f);sources[f'{layer}_{variant}']=dict(path=str(f),sha256=locked[str(f)],updates=record['updates'],nmse=record['nmse'])
    for f,h in locked.items():assert sha(f)==h,f
    atomic_json(ROOT/'parent_lock.json',locked);atomic_json(ROOT/'initial_sources.json',sources)
    atomic_json(ROOT/'preflight.json',dict(passed=True,reuses_exact_forward_and_gradient_validation=str(PARENT/'preflight.json'),new_model_math=False,initial_full_val_must_match_parent_best=True))
    print(ROOT)

if __name__=='__main__':main()
