"""Regression for layer27 FP16-gradient overflow; deployment forward is unchanged."""
import json
import random
import torch
from ..common import config, load_raw, paired_seed, atomic_json
from ..opt11.data import load_group, dense
from .model import HalfBmmFP32Backward, JointRouter


def main():
    root, m = config('streaming_sparse/v10/experiments/opt12_joint_20261025')
    paired_seed(12027)
    left = torch.randn(2, 4, 8, device='cuda', requires_grad=True)
    right = torch.randn(2, 8, 6, device='cuda', requires_grad=True)
    old = torch.bmm(left.half(), right.half()).float()
    repaired = HalfBmmFP32Backward.apply(left, right)
    assert torch.equal(old, repaired)
    upstream = torch.full_like(old, 90000.)
    old_grads = torch.autograd.grad(old, (left, right), upstream)
    new_grads = torch.autograd.grad(repaired, (left, right), upstream)
    expected = (torch.bmm(upstream, right.detach().half().float().transpose(1, 2)),
                torch.bmm(left.detach().half().float().transpose(1, 2), upstream))
    assert any(not torch.isfinite(g).all() for g in old_grads)
    assert all(torch.isfinite(g).all() for g in new_grads)
    assert all(torch.equal(g, e) for g, e in zip(new_grads, expected))
    chosen = random.Random(12027).sample([r for r in m['rows'] if r['split'] == 'train'], 4)
    records = []
    for variant in m['variants']:
        model = JointRouter(m['parent_experiment'], m['parent_opt11'], 27, variant, m['monitor_beta']).cuda()
        model.zero_grad(set_to_none=True)
        for row in chosen:
            q = load_group(m['parent_opt11'], row, 27)
            raw = load_raw(m['parent_experiment'], row, 27)
            truth = dense(q, raw).float()
            with torch.no_grad():
                expected_y, _, _ = model.deployed(q, raw, row['step'])
            y, keep, _ = model(q, raw, row['step'])
            assert torch.equal(y, expected_y)
            nmse = (y - truth).square().sum() / truth.square().sum()
            assert torch.equal(nmse, (expected_y - truth).square().sum() / truth.square().sum())
            loss = (nmse + m['budget_penalty'] * (keep - m['target_keep']).clamp_min(0).square()) / 4
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            records.append(dict(variant=variant, event=row['sample_id'], nmse=float(nmse), forward_equal=True))
        before = torch.nn.utils.clip_grad_norm_(model.routing.parameters(), 1., error_if_nonfinite=True)
        head_before = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1., error_if_nonfinite=True)
        after = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.routing.parameters() if p.grad is not None]))
        assert float(after) <= 1.00001
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-7, weight_decay=0.)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print(variant, 'PASS', 'pre_clip_norm',float(before),'post_clip_norm',float(after), flush=True)
    result = dict(passed=True, original_half_backward_overflow_reproduced=True,
                  bmm_forward_bit_identical=True, float32_jacobian_matches_reference=True,
                  real_failure_batch_all_gradients_finite=True, clipping_and_update_finite=True, events=records)
    atomic_json(root/'repairs/nonfinite_20261025/regression.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
