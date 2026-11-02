"""GPU preflight: unchanged dense semantics, true sparse kernel, and cache caps."""
import json
from pathlib import Path
import types
import torch

from .baseline_methods import BaselineAdapter, dense
from .kv_history import audit_attention, forward, compact_cache
from .anchor_rebase import AnchorMonitor


@torch.no_grad()
def main():
    from streaming_sparse.v08.audit import TinyWrapper
    from streaming_sparse.v08.backbone import native_cache
    from streaming_sparse.capture.capture_runner import sdpa_backend
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    w = TinyWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    w.model.num_frame_per_block = 3
    emb = torch.randn(1, 4, 32, device='cuda', dtype=torch.bfloat16)
    inputs = torch.randn(8, 1, 3, 16, 16, 16, device='cuda', dtype=torch.bfloat16)
    ft = 64
    checks = {}
    with sdpa_backend():
        for method, policy in [('fifo', 'B_latest_five'), ('anchor', 'A_first_chunk'), ('rebase', 'A_first_chunk')]:
            cache = native_cache(w, 'cuda', torch.bfloat16, 18 * ft)
            retained, expected = [], []
            with audit_attention() as audit:
                monitor = AnchorMonitor(w, audit, 'rebased' if method == 'rebase' else 'baseline', 8, 8)
                with monitor.install():
                    for chunk, noisy in enumerate(inputs):
                        for step, ts in enumerate([937.5, 0.]):
                            monitor.begin(chunk, step, 'clean_context' if step else 'denoise')
                            expected.append(forward(w, noisy, emb, ts, cache, chunk, ft, audit))
                        retained = compact_cache(cache, retained, chunk + 1, ft, policy)
                        monitor.flush()
            adapter = BaselineAdapter(w, method, ft)
            with adapter.install():
                for chunk, noisy in enumerate(inputs):
                    for step, ts in enumerate([937.5, 0.]):
                        adapter.begin(chunk, step, clean=bool(step))
                        actual = w(noisy, {'prompt_embeds': emb}, torch.full((1, 3), ts, device='cuda'),
                                   kv_cache=[{}] * 3, crossattn_cache=[None] * 3, current_start=chunk * 3 * ft)[1]
                        torch.testing.assert_close(actual, expected[chunk * 2 + step], atol=0, rtol=0)
                    adapter.end_chunk()
            checks[method + '_exact_vs_prior_implementation'] = True
        adapter = BaselineAdapter(w, 'lightforcing', ft)
        q = torch.randn(1, 3 * ft, 2, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 18 * ft, 2, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        smap, lut, topk = adapter.kernel.get_sm_80_120_block_map(q, k, .25, 64, 64, ft, 6, 3, 2)
        out = adapter.kernel._attention.apply(q, k, v, smap, lut, topk, 64, 64, None)
        mask = smap.bool().repeat_interleave(64, -2).repeat_interleave(64, -1)
        ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).float(), k.transpose(1,2).float(),
                    v.transpose(1,2).float(), attn_mask=mask).transpose(1,2)
        torch.testing.assert_close(out.float(), ref, atol=.007, rtol=.025)
        checks['lightforcing_sparse_vs_masked_fp32_max_error'] = float((out.float() - ref).abs().max())
        checks['lightforcing_selected_fraction'] = float(topk * 64 / k.shape[1])
        for method in ['lightforcing', 'dasa_kv']:
            adapter = BaselineAdapter(w, method, ft)
            with adapter.install():
                for chunk, noisy in enumerate(inputs):
                    for step, ts in enumerate([937.5, 0.]):
                        adapter.begin(chunk, step, clean=bool(step))
                        result = w(noisy, {'prompt_embeds': emb}, torch.full((1,3), ts, device='cuda'),
                            kv_cache=[{}]*3, crossattn_cache=[None]*3, current_start=chunk*3*ft)[1]
                        assert torch.isfinite(result).all()
                    assert all(c['k'].shape[1] <= 15*ft for c in adapter.cache)
            checks[method + '_eight_chunk_finite_and_budgeted'] = True
        from wan.modules.causal_model import CausalWanSelfAttention
        blocks = [types.SimpleNamespace(self_attn=CausalWanSelfAttention(1536, 12).cuda().to(torch.bfloat16)) for _ in range(3)]
        fake = types.SimpleNamespace(model=types.SimpleNamespace(blocks=blocks))
        adapter = BaselineAdapter(fake, 'forcingkv', 192)
        grid = torch.tensor([[3, 12, 16]])
        for chunk in range(3):
            for clean in [False, True]:
                adapter.begin(chunk, int(clean), clean)
                for layer, b in enumerate(blocks):
                    x = torch.randn(1, 576, 1536, device='cuda', dtype=torch.bfloat16)
                    result = adapter.forward(layer, b.self_attn, x, torch.tensor([576]), grid,
                                             w.model.freqs, None, current_start=chunk*576)
                    assert result.shape == x.shape and torch.isfinite(result).all()
            adapter.end_chunk()
        checks['forcingkv_official_grouped_cache_and_dynamic_selection'] = True
    result = dict(passed=True, checks=checks)
    path = Path('local_setup/results/baseline_methods_preflight.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
