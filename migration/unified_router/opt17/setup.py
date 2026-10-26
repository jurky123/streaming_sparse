import json
from pathlib import Path
from ..common import atomic_json,sha
ROOT=Path('streaming_sparse/v10/experiments/opt17_grouped_joint_20261025').resolve()
PARENT=Path('streaming_sparse/v10/experiments/opt16_grouped_router_20261025').resolve()
DATA=Path('streaming_sparse/v10/experiments/opt11_finetune_20261025').resolve()

def main():
    if (ROOT/'experiment.json').exists():print(ROOT);return
    parent=json.loads((DATA/'experiment.json').read_text())
    m=dict(parent,parent_opt11=str(DATA),parent_opt16=str(PARENT),layers=[1,14,24,29],
        variants={'Fast35':dict(candidate_fraction=.35,representatives=8),'Quality50':dict(candidate_fraction=.5,representatives=16)},
        full_query_cache=str(ROOT/'full_queries'),target_keep=.15,tree_depth=6,calibration_events=12,
        max_updates=2000,min_updates=1000,validate_every=100,checkpoint_every=25,batch_events=2,
        learning_rate_router=3e-5,warmup_updates=50,gate_temperature=.5,
        patience_validations=6,min_relative_improvement=1e-4,tree_refresh_every=250,tree_probe_events=6,
        trainable=['current/history prototype assignment weights and biases','key projection first64'],
        alternating=['Train-only calibrated Q/K grouping trees; proposed every250 steps; accepted using actual Train attention NMSE and executed tile constraint'],
        frozen=['backbone','inactive projection columns','value basis','final keep budget','INT8 QK/BF16 PV implementation'],
        training_objective='actual deployed hard attention output NMSE on sampled original queries, with full4680-query group/quantization context',
        backward='biased STE fine/coarse selection, original128d page-gate attention Jacobian; hard grouping updated by alternating Train proposal evaluation',
        best_metric='actual_deployment_attention_nmse',test_used=False,
        parallelism='8 GPUs: four pilot layers times two independent candidate-budget variants; not DDP',
        capture_query_design='full4680 queries, verified against existing raw Q; reuse existing full KV; Train192/Val96',
        tree_max_relative_tile_increase=1.05,tree_min_relative_improvement=1e-4,
        validation_extra_candidate_fractions=[.35,.5,.75])
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'experiment.json',m)
    locked=json.loads((PARENT/'code_manifest.json').read_text())
    for layer in m['layers']:
        f=Path(m['parent_experiment'])/f'action_basis/layer{layer:02d}.pt';locked[str(f)]=sha(f)
    for f,h in locked.items():assert sha(f)==h,f
    atomic_json(ROOT/'parent_lock.json',locked)
    print(ROOT)

if __name__=='__main__':main()
