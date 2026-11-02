"""Check the actual Opt28/30 native sparse kernel on synthetic GPU tensors."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from migration.unified_router.opt24.sparge import quant
from migration.unified_router.opt28.native import Native

torch.manual_seed(29)
torch.set_num_threads(4)
h, n, seg, g, k = 2, 128, 128, 64, 128
q = torch.randn(h, n, 128, device='cuda', dtype=torch.bfloat16) * 0.3
keys = torch.randn(h, 2 * seg, 128, device='cuda', dtype=torch.bfloat16) * 0.3
values = torch.randn_like(keys)
kv = SimpleNamespace(k=keys, v=values)
native = Native(kv, seg, g, k, 16)
ids = torch.arange(2, device='cuda', dtype=torch.int32)[None, None].expand(h, n // g, -1).contiguous()
order = torch.arange(n, device='cuda', dtype=torch.int32)[None].expand(h, -1).contiguous()
actual = native.prepared(q, ids, order)
qi, qs = quant(q, g)
q_reference = qi.float() * qs.repeat_interleave(g, dim=1)[..., None]
k_reference = native.ki.float() * native.ks.repeat_interleave(k, dim=1)[..., None]
expected = F.scaled_dot_product_attention(q_reference, k_reference, native.v.float())
torch.cuda.synchronize()
torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.02)
assert torch.isfinite(actual).all()
result = dict(passed=True, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
              cuda=torch.version.cuda, kernel='opt28_tiles_v2',
              max_abs_vs_quantized_dense=float((actual.float() - expected).abs().max()))
output = Path(__file__).resolve().parent / 'results'
output.mkdir(parents=True, exist_ok=True)
(output / 'cuda_backend.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
