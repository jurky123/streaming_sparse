"""Forward self-attention matrix work, including dense portions of the rollout."""
import ast
import math
import sys
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'reproduction/vbench_current'))
from quota_counts import counts

policy = np.load(repo / 'reproduction/vbench_current/policies.npz')['10_window_step_layer_head']
# Import the upstream scheduling function without loading CUDA-only kernels.
source = repo / 'external_baselines/LightForcing/wan/modules/sparse_attention.py'
scheduler = next(node for node in ast.parse(source.read_text()).body
                 if getattr(node, 'name', '') == 'calculate_chunk_sparsities')
scope = {'math': math}
exec(compile(ast.Module(body=[scheduler], type_ignores=[]), str(source), 'exec'), scope)
for chunks, target, initial in [(7, .88, .98), (41, .85, .95)]:
    sparsity = scope['calculate_chunk_sparsities'](
        chunks * 3, 3, 18, dict(sparsity=target, sparsity_base=initial))
    work = np.zeros(4)
    queries = 4608 * 12
    for c in range(chunks):
        pages = min(432, 72 * (c + 1))
        dense = queries * 4 * 128 * 64 * pages
        light_pages = int(pages * (1 - sparsity[c]))
        work[0] += 150 * dense
        work[1] += 150 * queries * 4 * 128 * 64 * light_pages
        if c < 2:
            work[2:] += 150 * dense
            continue
        quota = np.array(counts(pages)).sum(axis=1)
        exact_pages = np.array([pages * 15 // 100, 24 + quota[policy[c - 2]].mean()])
        # Same fused-operation convention as profiling/tier_audit/analyze.py.
        fused = queries * (6 * 128 * 64 * exact_pages + pages * 1632 + 2048)
        work[2:] += 34 * dense + 116 * fused
    print(chunks, 'chunks, TFLOP (dense, LightForcing, base, intersection)', work / 1e12)
