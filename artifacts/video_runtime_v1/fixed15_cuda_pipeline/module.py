"""Replay the bitwise CUDA candidate on one captured fixed-15% window."""
import ctypes
import json
from pathlib import Path
from statistics import median
import sys
import torch
from local_experiments.video_runtime_v1.page_prefetch import coeff_tail
from pack_kernel import pack_range
from summary_fragments import encode
from repair import retry

root = Path(__file__).parent
sys.path.insert(0, str(root.parent / 'fixed15_baseline_v1'))
import bench as fixture

q, k, v = fixture.q, fixture.k, fixture.v
ids, basis = fixture.ids, fixture.basis
keys, mean, detail = fixture.keys, fixture.mean, fixture.detail
p, hp, count = fixture.p, fixture.hp, fixture.count
expected, actual = torch.empty_like(q), torch.empty_like(q)
packed_k, packed_v = torch.empty_like(k), torch.empty_like(v)
tiles = (p + 15) // 16
summary = torch.empty((12, tiles, 10624), dtype=q.dtype, device=q.device)
flags = torch.empty((12, 72), dtype=torch.int32, device=q.device)
lib = ctypes.CDLL('/tmp/jzk_fixed15_paired_k.so')
lib.run.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 2 + [ctypes.c_void_p]

def reference():
    return coeff_tail[(72,12)](q,keys,mean,detail,None,None,None,None,p,4608,12,128,
        HEAD_PAGES=hp,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=expected,
        K=k,V=v,IDS=ids,EXACT_COUNT=count,QK_SCALE=128**-.5,ANCHOR_SCALE=True,
        COLUMN_MASS=True,num_warps=4,num_stages=2,enable_fp_fusion=False)

def pack(start):
    pack_range[(p-start,12,16)](k,v,packed_k,packed_v,start)

def fused():
    args = (q, packed_k, packed_v, ids, actual, summary, basis, flags)
    status = lib.run(*(x.data_ptr() for x in args), p, count, torch.cuda.current_stream().cuda_stream)
    assert status == 0

def candidate():
    pack(p-72)
    encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,num_warps=4)
    fused()
    retry[(72,12)](flags,q,k,v,ids,keys,mean,detail,basis,actual,p,hp,count,
        num_warps=4,num_stages=2,enable_fp_fusion=False)

pack(0)
assert '%waitv =' in reference().asm['ttgir']
candidate()
assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
times = fixture.times([fixture.timed(f) for f in (reference,candidate,fused)])
row = dict(pages=p,exact_pages=count,bitwise=True,ms=list(map(median,times)),
           speedup=median(a/b for a,b in zip(*times[:2])),samples=times)
(root/f'confirmed_{p}.json').write_text(json.dumps(row,indent=2))
print(row['pages'],row['ms'],row['speedup'])
