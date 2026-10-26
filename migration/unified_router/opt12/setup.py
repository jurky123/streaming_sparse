"""Create an isolated experiment; leave Opt10/Opt11 sealed."""
import json
from pathlib import Path
from ..common import atomic_json, sha

PARENT = Path('streaming_sparse/v10/experiments/opt11_finetune_20261025').resolve()
ROOT = Path('streaming_sparse/v10/experiments/opt12_joint_20261025').resolve()


def main():
    if (ROOT / 'experiment.json').exists():
        print(ROOT)
        return
    parent = json.loads((PARENT / 'experiment.json').read_text())
    m = dict(parent, parent_opt11=str(PARENT), training_objective='actual_current_deployed_hard_attention_nmse',
             backward='STE selection plus original128d page-gate attention Jacobian; biased surrogate, not exact quantization/Skip0 derivative',
             initialization='Opt11 strong-stage best heads and unchanged parent prototype parameters',
             max_updates=2000, min_updates=1000, validate_every=100, checkpoint_every=25,
             patience_validations=6, min_relative_improvement=1e-4, batch_events=4,
             learning_rate_head=3e-4, learning_rate_router=3e-5, warmup_updates=50,
             gate_temperature=.25, action_temperature=.25, target_keep=.15, max_best_keep=.16,
             budget_penalty=10., group_cache=parent['group_cache'], feature_cache=None,
             counterfactual=None, trainable=['current_weight', 'current_bias', 'history_8_weight',
                'history_8_bias', 'history_4_weight', 'history_4_bias', 'history_2_weight',
                'history_2_bias', 'key_basis_first64', 'action_head'],
             frozen=['backbone_QKV', 'inactive_key_basis_tail', 'unused_value_basis',
                     'feature_normalization', 'beta', 'quantization_and_skip_rules'],
             parallelism='8 GPUs; disjoint layers per worker; both variants per layer; not replicated DDP',
             logging='append-only per-step JSONL train NMSE, budget, entropy, gradient norms; raw full validation every100',
             test_used=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / 'experiment.json', m)
    frozen = {}
    for f in (Path('migration/unified_router/opt11')).glob('*.py'):
        frozen[str(f.resolve())] = sha(f)
    for layer in m['layers']:
        for variant in m['variants']:
            f = PARENT / f'strong_training/layer{layer:02d}/{variant}/best.pt'
            frozen[str(f)] = sha(f)
    lock = json.loads((PARENT / 'parent_lock.json').read_text())
    frozen.update(lock['source'])
    frozen.update(lock['parameters'])
    for f, digest in frozen.items():
        assert sha(f) == digest, f
    atomic_json(ROOT / 'parent_lock.json', frozen)
    print(ROOT)


if __name__ == '__main__':
    main()
