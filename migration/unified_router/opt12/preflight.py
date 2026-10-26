"""Real long-history input: exact forwards, gradient coverage and state updates."""
import argparse, json, time
from pathlib import Path
import torch
from ..common import config, load_raw, atomic_json, paired_seed
from ..opt11.data import Builder, load_group, dense, attend
from ..opt11.head import apply, exact
from .model import JointRouter, ActualAttention


def check(root, m):
    paired_seed(12)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row = next(r for r in m['rows'] if r['split'] == 'train' and r['history_tokens'] == 14040)
    layer = 1
    raw = load_raw(m['parent_experiment'], row, layer)
    q = load_group(m['parent_opt11'], row, layer)
    truth = dense(q, raw).float()
    builder = Builder(m['parent_experiment'], layer, m['monitor_beta'])
    with torch.no_grad():
        old_d, _ = builder.inputs(q, raw, row['step'])
    records = {}
    for variant in m['variants']:
        model = JointRouter(m['parent_experiment'], m['parent_opt11'], layer, variant, m['monitor_beta']).cuda()
        with torch.no_grad():
            old_mask = apply(old_d, exact(model.head, old_d).argmax(-1))
            old_y = attend(variant, q, raw, old_mask).float()
            y, mask, d = model.deployed(q, raw, row['step'])
            assert torch.equal(old_mask, mask), variant
            assert torch.equal(y, old_y), variant
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.)
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        start = time.perf_counter()
        actual, keep, entropy = model(q, raw, row['step'])
        assert torch.equal(actual, y), variant
        loss = (actual - truth).square().sum() / truth.square().sum()
        reference = (y - truth).square().sum() / truth.square().sum()
        assert torch.equal(loss, reference)
        loss.backward()
        grads = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.named_parameters()}
        assert all(v is not None for v in grads.values()), grads
        assert all(v > 0 for n, v in grads.items() if n.startswith('routing.')), grads
        assert sum(v for n, v in grads.items() if n.startswith('head.')) > 0, grads
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        changed = {n: not torch.equal(p, initial[n]) for n, p in model.named_parameters()}
        assert all(v for n, v in changed.items() if n.startswith('routing.')), changed
        assert any(v for n, v in changed.items() if n.startswith('head.')), changed
        with torch.no_grad():
            next_y, next_mask, next_d = model.deployed(q, raw, row['step'])
            # Changing prototypes must update routing, not cached fixed inputs.
            assert not torch.equal(next_d['score'], d['score'])
        torch.cuda.synchronize()
        records[variant] = dict(forward_bit_identical=True, actual_loss_bit_identical=True,
            all_parameter_groups_have_finite_nonzero_gradient=True, gradients=grads, changed=changed,
            nmse=float(loss), keep=float(keep), entropy=float(entropy),
            seconds=time.perf_counter()-start, peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        del model, actual, next_y, optimizer
    # Validate the implemented page-gate Jacobian against Torch autograd.
    z = torch.randn(2, 3, 5, device='cuda')
    v = torch.randn(2, 3, 5, 7, device='cuda')
    gate = torch.rand(2, 3, 5, device='cuda', requires_grad=True)
    mass = z.softmax(-1)
    expected = ((mass * gate)[..., None] * v).sum(-2) / (mass * gate).sum(-1, keepdim=True)
    upstream = torch.randn_like(expected)
    reference_grad, = torch.autograd.grad(expected, gate, upstream)
    actual = ActualAttention.apply(gate, expected.detach(), z, v)
    our_grad, = torch.autograd.grad(actual, gate, upstream)
    assert torch.allclose(our_grad, reference_grad, atol=2e-6, rtol=2e-5)
    result = dict(passed=True, variant_checks=records, attention_surrogate_jacobian_checked=True,
                  event=row['sample_id'], layer=layer, raw_history_tokens=row['history_tokens'])
    atomic_json(root / 'preflight.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    a = p.parse_args()
    root, m = config(a.directory)
    check(root, m)


if __name__ == '__main__':
    main()
