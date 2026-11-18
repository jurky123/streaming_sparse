"""Check that a masked representative cannot contaminate its selecting query block."""
import ctypes
import json
import sys
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,ids,summary,basis,p,hp,keys,mean,detail,implementation
tiles=(p+15)//16
def pack():
    encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
def query(out):return out[0,:64,0].contiguous()
pack()
base,a,flags=implementation(library())
cand,b,bad=implementation(ctypes.CDLL('/tmp/jzk_page_maskbias.so'))
ref=query(a).clone()
page=int(ids[0,0,0])
assert page>=32
keys[0,page].fill_(float('nan'))
pack()
base();cand()
a,b=query(a),query(b)
row=dict(page=page,baseline_bitwise=torch.equal(ref.view(torch.uint8),a.view(torch.uint8)),candidate_nan=int(torch.isnan(b).sum()),elements=b.numel(),baseline_flag=int(flags[0,0]),candidate_flag=int(bad[0,0]))
assert row['baseline_bitwise']
(root/'masked_nonfinite.json').write_text(json.dumps(row,indent=2))
print(row)
