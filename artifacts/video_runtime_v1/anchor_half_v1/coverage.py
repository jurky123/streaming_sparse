"""Count additional entirely masked representative groups on captured exact routes."""
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
result={}
for pages in (216,288,360,432):
    ids=torch.load(f'/tmp/jzk_fixed15_input_{pages}.pt',weights_only=True,map_location='cpu')['ids']
    mask=torch.zeros((*ids.shape[:-1],pages),dtype=torch.bool)
    mask.scatter_(-1,ids.long(),True)
    assert mask[...,:24].all()
    extra=mask[...,24:].reshape(*ids.shape[:-1],-1,8).all(-1)
    padding=(-pages)%16
    rounded=torch.nn.functional.pad(mask,(0,padding),value=True)
    tiles=rounded[...,16:].reshape(*ids.shape[:-1],-1,16).all(-1)
    result[pages]=dict(query_head_tiles=extra[...,0].numel(),groups=extra.numel(),
        fully_masked_tiles=int(tiles.sum()),total_tiles=tiles.numel(),all_selected=int(extra.sum()),tiles_with_extra=int(extra.any(-1).sum()))
Path(__file__).with_name('coverage.json').write_text(json.dumps(result,indent=2)+'\n')
