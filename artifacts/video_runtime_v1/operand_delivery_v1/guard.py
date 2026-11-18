"""Compare tile fallback decisions and exercise the complete retry path."""
import json
import os
import torch
import ctypes as C
from capture import basis, count, detail, hp, ids, implementation, k, keys, mean, p, pack_range, pk, pv, q, root, v

from local_experiments.video_runtime_v1.fixed15_layout import retry
def check():
    old,a,af=implementation(C.CDLL('/tmp/jzk_delivery_control.so'))
    new,b,bf=implementation(C.CDLL('/tmp/jzk_delivery_'+os.environ['CANDIDATE']+'.so'))
    assert torch.equal(af,bf)
    for out,flags in ((a,af),(b,bf)):
        retry[(72,12)](flags,q,k,v,ids,keys,mean,detail,basis,out,p,hp,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    if af.all() or os.environ['CANDIDATE']=='late_mask':
        assert torch.equal(a,b)
    return int(af.sum())

normal=check()
k[:,:,0].zero_();k[:,:64,0].fill_(10);q[:,:64,0].fill_(10)
pack_range[(p,12,16)](k,v,pk,pv,0)
partial=check()
k.zero_();k[:,:64].fill_(10);q.fill_(10)
pack_range[(p,12,16)](k,v,pk,pv,0)
extreme=check()
assert normal==0 and 0<partial<864 and extreme==864,(normal,partial,extreme)
(root/f"{os.environ['CANDIDATE']}_guard_{p}.json").write_text(json.dumps(dict(p=p,normal=normal,partial=partial,extreme=extreme,all_fallback_bitwise=True,flags_match=True)))
