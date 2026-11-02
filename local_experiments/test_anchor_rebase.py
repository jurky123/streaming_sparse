"""Numerical checks for temporal-only shifts, telemetry and native-model equivalence."""
import argparse
import json
from pathlib import Path

import torch

from .anchor_rebase import shift_temporal_, anchor_mass, AnchorMonitor
from .kv_history import audit_attention, forward, compact_cache


@torch.no_grad()
def checks():
    from wan.modules.model import rope_params
    from wan.modules.causal_model import causal_rope_apply
    from streaming_sparse.v08.audit import TinyWrapper
    from streaming_sparse.v08.backbone import native_cache
    from streaming_sparse.capture.capture_runner import sdpa_backend
    torch.manual_seed(41)
    freqs = torch.cat([rope_params(1024, 44), rope_params(1024, 42), rope_params(1024, 42)], dim=1).cuda()
    raw = torch.randn(1, 12, 2, 128, device='cuda', dtype=torch.float64)
    grid = torch.tensor([[3, 2, 2]])
    canonical = causal_rope_apply(raw, grid, freqs)
    errors = []
    for delta in (0, 3, 60, 105):
        result = shift_temporal_(canonical.clone(), 12, freqs, delta)
        expected = causal_rope_apply(raw, grid, freqs, start_frame=delta)
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)
        assert torch.equal(result[..., 44:], canonical[..., 44:])
        errors.append(float((result - expected).abs().max()))
    bf16 = canonical.bfloat16()
    assert torch.equal(shift_temporal_(bf16.clone(), 12, freqs, 0), bf16)
    q = torch.randn(1, 12, 2, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 24, 2, 128, device='cuda', dtype=torch.bfloat16)
    alternate = shift_temporal_(k[:, :12].clone(), 12, freqs, 60)
    indices = torch.arange(12, device='cuda')
    measured, cf = anchor_mass(q, k, 12, indices, alternate)
    direct, _ = anchor_mass(q, torch.cat([alternate, k[:, 12:]], dim=1), 12, indices)
    torch.testing.assert_close(cf, direct, atol=1e-6, rtol=1e-6)
    logits = q[0].transpose(0, 1).float() @ k[0].permute(1, 2, 0).float() / 128**0.5
    torch.testing.assert_close(measured, logits.softmax(-1)[..., :12].sum(-1).mean(-1), atol=1e-6, rtol=1e-6)
    wrapper = TinyWrapper().cuda().bfloat16().eval().requires_grad_(False)
    wrapper.model.num_frame_per_block = 3
    embeds = torch.randn(1, 4, 32, device='cuda', dtype=torch.bfloat16)
    inputs = torch.randn(8, 1, 3, 16, 8, 8, device='cuda', dtype=torch.bfloat16)
    outputs = {}
    for variant in ('native', 'baseline', 'rebased'):
        cache = native_cache(wrapper, 'cuda', torch.bfloat16, tokens=18 * 16)
        retained, trajectory = [], []
        from contextlib import nullcontext
        with sdpa_backend(), audit_attention() as audit:
            monitor = AnchorMonitor(wrapper, audit, variant, 4, 4) if variant != 'native' else None
            with monitor.install() if monitor else nullcontext():
                for chunk in range(8):
                    current = inputs[chunk]
                    for step, timestep in enumerate((937.5, 625.0, 0.0)):
                        if monitor:
                            monitor.begin(chunk, step, 'clean_context' if timestep == 0 else 'denoise')
                        current = forward(wrapper, current, embeds, timestep, cache, chunk, 16, audit)
                    trajectory.append(current.cpu())
                    # Before the first rebase affects any output, canonical cached K/V match exactly.
                    if chunk == 5:
                        snapshot = [(x['k'].clone(), x['v'].clone()) for x in cache]
                        if variant == 'native':
                            native_snapshot = snapshot
                        else:
                            for pair, reference in zip(snapshot, native_snapshot):
                                assert all(torch.equal(x, y) for x, y in zip(pair, reference))
                    retained = compact_cache(cache, retained, chunk + 1, 16, 'A_first_chunk')
                    if monitor:
                        monitor.flush()
        outputs[variant] = torch.cat(trajectory, 1)
    assert torch.equal(outputs['native'], outputs['baseline'])
    assert torch.equal(outputs['baseline'][:, :18], outputs['rebased'][:, :18])
    assert not torch.equal(outputs['baseline'][:, 18:], outputs['rebased'][:, 18:])
    return dict(passed=True, temporal_shift_max_abs=max(errors), spatial_components_bitwise_unchanged=True,
                zero_shift_bf16_bitwise_identical=True, full_key_softmax_mass_verified=True,
                same_state_counterfactual_mass_verified=True, instrumented_native_outputs_exact=True,
                first_six_chunks_and_canonical_cache_exact=True, rebase_changes_later_outputs=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = checks()
    target = Path(a.output)
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
