"""Check equal budgets and compare compact KV to an independent full-history archive."""
import argparse
import json
from pathlib import Path

import torch

from .kv_history import POLICIES, select_frames, compact_cache, audit_attention, forward


def policy_checks():
    for policy in POLICIES:
        old = []
        for completed in range(1, 43):
            chosen = select_frames(policy, completed)
            assert len(chosen) == min(15, completed * 3)
            assert chosen == sorted(set(chosen))
            assert set(chosen) <= set(old + list(range((completed - 1) * 3, completed * 3)))
            old = chosen
    assert select_frames('A_first_chunk', 20) == list(range(3)) + list(range(48, 60))
    assert select_frames('B_latest_five', 20) == list(range(45, 60))
    assert select_frames('C_first_three_starts', 20) == [0, 3, 6] + list(range(48, 60))
    for completed in range(6):
        assert len({tuple(select_frames(p, completed)) for p in POLICIES}) == 1


@torch.no_grad()
def gpu_checks():
    from streaming_sparse.v08.audit import TinyWrapper
    from streaming_sparse.v08.backbone import native_cache
    from streaming_sparse.capture.capture_runner import sdpa_backend
    torch.manual_seed(29)
    wrapper = TinyWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    wrapper.model.num_frame_per_block = 3
    frame_tokens = 16
    embeds = torch.randn(1, 4, 32, device='cuda', dtype=torch.bfloat16)
    inputs = torch.randn(8, 1, 3, 16, 8, 8, device='cuda', dtype=torch.bfloat16)
    rows = []
    with sdpa_backend(), audit_attention() as audit:
        for policy in POLICIES:
            actual = native_cache(wrapper, 'cuda', torch.bfloat16, tokens=18 * frame_tokens)
            oracle = native_cache(wrapper, 'cuda', torch.bfloat16, tokens=18 * frame_tokens)
            archive = [dict(k=x['k'].new_empty(1, 24 * frame_tokens, *x['k'].shape[2:]),
                            v=x['v'].new_empty(1, 24 * frame_tokens, *x['v'].shape[2:])) for x in actual]
            retained = []
            for chunk in range(8):
                ids = select_frames(policy, chunk)
                # This reference retrieves every retained row from a non-evicting archive.
                token_ids = [f * frame_tokens + t for f in ids for t in range(frame_tokens)]
                index = torch.tensor(token_ids, device='cuda', dtype=torch.long)
                h = len(token_ids)
                for ref, all_rows in zip(oracle, archive):
                    for kind in ('k', 'v'):
                        ref[kind][:, :h].copy_(all_rows[kind].index_select(1, index))
                    ref['local_end_index'].fill_(h)
                    ref['global_end_index'].fill_(chunk * 3 * frame_tokens)
                current = inputs[chunk]
                for ts in (937.5, 625.0, 0.0):
                    a = forward(wrapper, current, embeds, ts, actual, chunk, frame_tokens, audit)
                    b = forward(wrapper, current, embeds, ts, oracle, chunk, frame_tokens, audit)
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                    current = a
                for ref, all_rows in zip(oracle, archive):
                    for kind in ('k', 'v'):
                        all_rows[kind][:, chunk * 3 * frame_tokens:(chunk + 1) * 3 * frame_tokens].copy_(
                            ref[kind][:, h:h + 3 * frame_tokens])
                retained = compact_cache(actual, retained, chunk + 1, frame_tokens, policy)
            rows.append(dict(policy=policy, chunks=8, exact_vs_archive=True))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy_checks()
    result = dict(passed=True, budget_checks=True, gpu=gpu_checks())
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
