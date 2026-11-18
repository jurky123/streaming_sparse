"""Measure whole and partial compensation tiles after the permanent prefix."""
import json
from pathlib import Path
import torch

report={}
for p in (216,288,360,432):
    ids=torch.load(f'/tmp/jzk_fixed15_input_{p}.pt',weights_only=True)['ids'].reshape(864,-1)
    selected=torch.ones(864,(p+15)//16*16,dtype=torch.bool)
    selected[:,:p]=False
    selected.scatter_(1,ids.long(),True)
    levels={}
    for size in (4,8,16):
        counts=selected[:,16:].reshape(864,-1,size).sum(-1)
        levels[size]=dict(total=counts.numel(),fully_selected=int((counts==size).sum()),
                          empty=int((counts==0).sum()),partly_selected=int(((counts>0)&(counts<size)).sum()))
    report[p]=levels
(Path(__file__).parent/'coverage.json').write_text(json.dumps(report,indent=2))
print(report)
