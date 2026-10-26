"""Mathematical and amended-boundary checks before real-data routing."""
import argparse
import math
import torch
from .common import config, paired_seed, atomic_json, sha
from .core import build, scores, initial_mask, candidates, features, LabelStore, apply_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    args = p.parse_args()
    root, m = config(args.directory)
    torch.cuda.set_device(0)
    paired_seed(410)
    h, nq, d = 2, 7, 128
    parameter = dict(current_weight=torch.randn(h, 16, d, device='cuda')*.05,
        current_bias=torch.zeros(h, 16, device='cuda'),
        key_basis=torch.linalg.qr(torch.randn(h, d, d, device='cuda'))[0],
        value_basis=torch.linalg.qr(torch.randn(h, d, d, device='cuda'))[0][..., :32])
    for prototypes in [2, 4, 8]:
        parameter[f'history_{prototypes}_weight'] = torch.randn(h, prototypes, d, device='cuda')*.05
        parameter[f'history_{prototypes}_bias'] = torch.zeros(h, prototypes, device='cuda')
    checks = []
    for nh in [0, 1, 65, 129, 9360, 14040]:
        q = torch.randn(h, nq, d, device='cuda')*.3
        kh = torch.randn(h, nh, d, device='cuda')*.3
        vh = torch.randn_like(kh)
        kc = torch.randn(h, 67, d, device='cuda')*.3
        vc = torch.randn_like(kc)
        sketch = build(kh, vh, kc, vc, parameter)
        score, content = scores(q, sketch)
        mask = initial_mask(score, sketch, .5)
        if nh == 0:
            assert mask.all(), 'User amendment: first chunk must be Dense'
        torch.testing.assert_close(mask, initial_mask(score*3+7, sketch, .5), atol=0, rtol=0)
        flat = initial_mask(torch.zeros_like(score), sketch, 1.)
        assert flat.all()
        expected = [8 if a == 1 else 4 if a == 2 else 2 for a in range(math.ceil(nh/4680), 0, -1)] + [16]
        assert [p.count.shape[-1] for p in sketch.parts] == expected
        assert (sketch.current.key[..., 0, :] - sketch.current.key[..., 1, :]).abs().max() > 0
        actions = candidates(score, mask, sketch.origins, .5)
        if nh == 0:
            assert not actions['valid'][:, 1:].any()
        geometry, scalar = features(q, sketch, score, mask, actions, .5, 2, content)
        shifted = build(kh, vh+3.2, kc, vc+3.2, parameter)
        shifted_score, shifted_content = scores(q, shifted)
        shifted_geometry, _ = features(q, shifted, shifted_score, mask, actions, .5, 2, shifted_content)
        torch.testing.assert_close(geometry, shifted_geometry, atol=2e-5, rtol=2e-4)
        _, key_content = scores(q, sketch, content='K32')
        kg, _ = features(q, sketch, score, mask, actions, .5, 2, key_content)
        _, key_content2 = scores(q, shifted, content='K32')
        kg2, _ = features(q, shifted, shifted_score, mask, actions, .5, 2, key_content2)
        torch.testing.assert_close(kg, kg2, atol=0, rtol=0)
        truth = LabelStore(q, kh, vh, kc, vc)
        ll = torch.bmm(q.float(), torch.cat((kh, kc), 1).float().transpose(1, 2))/math.sqrt(d)
        dense = torch.bmm(ll.double().softmax(-1), torch.cat((vh, vc), 1).double())
        torch.testing.assert_close(truth.target, dense, atol=2e-12, rtol=2e-12)
        labels, base_error, _ = truth.action_labels(mask, actions)
        for index in [0, 1, 9, 17, 80]:
            valid = actions['valid'][:, index]
            choice = torch.where(valid, torch.full_like(valid, index, dtype=torch.long), torch.zeros_like(valid, dtype=torch.long))
            selected = apply_action(mask, actions, choice)
            token = selected.repeat_interleave(sketch.costs.long(), dim=-1)
            raw = torch.bmm(ll.double().masked_fill(~token, -torch.inf).softmax(-1), torch.cat((vh, vc), 1).double())
            torch.testing.assert_close(raw, truth.output(selected), atol=2e-12, rtol=2e-12)
            delta = raw-truth.output(mask)
            e0 = truth.output(mask)-dense
            g = -2*(e0*delta).sum(-1)-delta.square().sum(-1)
            torch.testing.assert_close(labels[..., index][valid.reshape(h, nq)], g[valid.reshape(h, nq)].float(), atol=2e-7, rtol=2e-5)
        checks.append(dict(history=nh, raw_output_max_abs=float((truth.target-dense).abs().max()),
            current_prototypes=16, history_prototypes=expected[:-1], first_chunk_dense=nh == 0,
            first_chunk_followup_anchor=False, online_features=24))
    atomic_json(root/'P0_math.json', dict(passed=True, checks=checks,
        core_sha256=sha(__file__.replace('audit.py', 'core.py')), audit_sha256=sha(__file__),
        no_online_true_mass=True, no_mask_ste=True, synthetic_only=True))
    print(checks, flush=True)


if __name__ == '__main__':
    main()
