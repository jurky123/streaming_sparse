"""Compare tile fallback decisions and exercise the complete retry path."""
import json
import os
import torch
import ctypes as C
from pathlib import Path
import sys
from local_experiments.video_runtime_v1.fixed15_attention import library
directory=Path(__file__).parent
sys.path.insert(0,str(directory.parent/'operand_delivery_v1'))
from capture import basis, count, detail, hp, ids, implementation, k, keys, mean, p, pack_range, pk, pv, q, root, v

from local_experiments.video_runtime_v1.fixed15_layout import retry
def check():
    old,a,af=implementation(C.CDLL('/tmp/jzk_delivery_floor_center.so'))
    new,b,bf=implementation(library())
    assert torch.equal(af,bf)
    for out,flags in ((a,af),(b,bf)):
        retry[(72,12)](flags,q,k,v,ids,keys,mean,detail,basis,out,p,hp,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    return int(af.sum())

normal=check()
k[:,:,0].zero_();k[:,:64,0].fill_(10);q[:,:64,0].fill_(10)
pack_range[(p,12,16)](k,v,pk,pv,0)
partial=check()
k.zero_();k[:,:64].fill_(10);q.fill_(10)
pack_range[(p,12,16)](k,v,pk,pv,0)
extreme=check()
assert normal==0 and 0<partial<864 and extreme==864,(normal,partial,extreme)

ids.copy_(torch.arange(count,device=q.device).expand_as(ids))
q.fill_(1)
k.zero_()
k[:,0].fill_(18)
pack_range[(p,12,16)](k,v,pk,pv,0)
dominant=check()
assert dominant==0
(directory/f'guard_{p}.json').write_text(json.dumps(dict(normal=normal,partial=partial,extreme=extreme,dominant=dominant,bitwise=True)))
