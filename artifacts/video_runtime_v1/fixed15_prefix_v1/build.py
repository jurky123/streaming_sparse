"""Specialize only the deterministic segments of the fixed-15% page lists."""
import inspect
from pathlib import Path
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.gap_exact_round_kernel import _anchored_exact
root = Path(__file__).parent
exact = inspect.getsource(_anchored_exact.fn)
loop = exact[exact.index('    for i in tl.range'):exact.index('    if COLUMN_MASS:\n        total = tl.sum(column_mass')]
prefix = loop.replace('tl.range(1, count)', 'tl.range(0, 24)').replace('tl.load(ids + i).to(tl.int32)', 'i')
exact = exact.replace(loop, prefix + loop.replace('tl.range(1, count)','tl.range(25, count)'))
tail = inspect.getsource(_tail.fn).replace('COLUMN_MASS:tl.constexpr=False):','COLUMN_MASS:tl.constexpr=False,PREFIX:tl.constexpr=False,SKIP:tl.constexpr=False):')
line = next(line for line in tail.splitlines() if '=_anchored_exact(' in line)
tail = tail.replace(line,'   if PREFIX:\n '+line.replace('_anchored_exact(', 'prefix_exact(')+'\n   else:\n '+line)
tail = tail.replace('range(tr.cdiv(P*4,BR))','range(1 if SKIP else 0,tr.cdiv(P*4,BR))')
imports = """import math
import triton as tr
import triton
import triton.language as tl
from local_experiments.review_v1.tuned_attention import _av, _attn_fwd, _anchored_exact
from local_experiments.gap_exact_round_kernel import _anchor_logits, _anchor_av
"""
(root/'kernel.py').write_text(imports+exact.replace('def _anchored_exact','def prefix_exact')+'\n'+tail.replace('def _tail','def specialized'))
