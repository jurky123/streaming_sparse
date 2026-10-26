"""Replay both variants at the exact layer28 state/batch that failed at update22."""
import json
import torch
from ..common import config, load_raw, paired_seed, atomic_json
from ..opt11.data import load_group, dense
from ..opt11.head import surrogate as old_surrogate
from .model import JointRouter
from .stable_head import surrogate


def main():
    root, m = config('streaming_sparse/v10/experiments/opt12_joint_20261025')
    paired_seed(12028)
    folder = root / 'repairs/nonfinite_20261025/head_followup'
    state = torch.load(folder / 'failing_state.pt', map_location='cpu', weights_only=True)
    records = []
    for variant in m['variants']:
        model = JointRouter(m['parent_experiment'], m['parent_opt11'], 28, variant, m['monitor_beta']).cuda()
        model.load_state_dict(state['models'][variant])
        for row in state['rows']:
            q = load_group(m['parent_opt11'], row, 28)
            raw = load_raw(m['parent_experiment'], row, 28)
            truth = dense(q, raw).float()
            with torch.no_grad():
                expected_y, _, d = model.deployed(q, raw, row['step'])
                old_logits = old_surrogate(model.head, d)
                new_logits = surrogate(model.head, d)
                assert torch.equal(old_logits, new_logits), (variant, row['sample_id'])
            y, keep, _ = model(q, raw, row['step'])
            assert torch.equal(y, expected_y)
            nmse = (y - truth).square().sum() / truth.square().sum()
            assert torch.equal(nmse, (expected_y - truth).square().sum() / truth.square().sum())
            ((nmse + 10 * (keep - .15).clamp_min(0).square()) / 4).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            records.append(dict(variant=variant, event=row['sample_id'], nmse=float(nmse),
                                head_surrogate_forward_bit_identical=True, actual_forward_bit_identical=True))
        rg = torch.nn.utils.clip_grad_norm_(model.routing.parameters(), 1., error_if_nonfinite=True)
        hg = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1., error_if_nonfinite=True)
        torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.).step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print(variant, 'PASS', float(rg), float(hg), flush=True)
    atomic_json(folder / 'regression.json', dict(passed=True, layer=28, failed_update=22,
                revision='fp32_all_surrogate_backward_v3', gradients_clipping_and_updates_finite=True, records=records))


if __name__ == '__main__':
    main()
