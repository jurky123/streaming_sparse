"""Count distinct current pages used by exact attention within a capture."""
from pathlib import Path
import torch
import json
torch.set_num_threads(2)
result={}
for p in (216,288,360,432):
    ids=torch.load(f'/tmp/jzk_fixed15_input_{p}.pt',weights_only=True,map_location='cpu')['ids']
    current=ids>=p-72
    counts=current.sum(-1)
    # Histogram by head over the current 72 pages.
    hist=torch.zeros((12,72),dtype=torch.long)
    ix=(ids-(p-72)).clamp_min(0).long().reshape(12,-1)
    hist.scatter_add_(1,ix,current.long().reshape(12,-1))
    result[p]=dict(mean_current_pages_per_query_tile=float(counts.float().mean()),min=int(counts.min()),
                   max=int(counts.max()),share=float(current.float().mean()),
                   distinct_current_pages_per_head=(hist>0).sum(-1).tolist(),
                   mean_query_tiles_per_current_page=float(hist.float().mean()))
Path(__file__).with_name('current_routes.json').write_text(json.dumps(result,indent=2)+'\n')
