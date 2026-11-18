"""Exercise a row with one dominant key and unit mass."""
import json
import torch
import ctypes as C
from capture import count, ids, implementation, k, p, pack_range, pk, pv, q, root, v

# Page zero contributes a single key whose rounded scaled maximum exceeds the
# exact product by 2^-17. All other exp2 values underflow at this scale.
ids.copy_(torch.arange(count,device=q.device).expand_as(ids))
q.fill_(1)
k.zero_()
k[:,0].fill_(18)
pack_range[(p,12,16)](k,v,pk,pv,0)
report={}
for name in ('control','fused_center','anchor_center','floor_center'):
    _,_,flags=implementation(C.CDLL('/tmp/jzk_delivery_'+name+'.so'))
    report[name]=int(flags.sum())
assert report==dict(control=0,fused_center=864,anchor_center=0,floor_center=0),report
(root/'dominant.json').write_text(json.dumps(report,indent=2))
