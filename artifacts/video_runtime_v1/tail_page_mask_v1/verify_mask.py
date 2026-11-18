"""Check the lane-aligned predicate against every captured query block and page."""
import json
from pathlib import Path

import torch

result={}
for p in (216,288,360,432):
    ids=torch.load(f'/tmp/jzk_fixed15_input_{p}.pt',weights_only=True)['ids'].reshape(864,-1)
    selected=torch.zeros(864,512,dtype=torch.bool)
    selected.scatter_(1,ids.long(),True)
    padded=selected.clone()
    if p%32:
        padded[:,p:p+32-p%32]=True
    weights=1<<torch.arange(32,dtype=torch.int64)
    words=(padded.reshape(864,16,32)*weights).sum(-1)
    checks=0
    for pg in range(16,p,16):
        for half_lane in (0,1):
            lane_bits=words[:,pg//32]>>((pg%32)+half_lane)
            for frag in range(8):
                index=pg+2*frag+half_lane
                before=selected[:,index] | (index>=p)
                after=(lane_bits>>(2*frag))&1
                assert torch.equal(before,after.bool())
                checks+=len(before)
    result[p]=dict(comparisons=checks,padding_pages=(-p)%16)
(Path(__file__).parent/'mask_contract.json').write_text(json.dumps(result,indent=2))
