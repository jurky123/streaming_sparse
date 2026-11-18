"""Stress duplicate risks, including ties at the capacity cutoff."""
import json
import torch
from .weights import ROOT
from .selected_id_order import tie_order

def main():
    torch.manual_seed(701)
    rows=[]
    for p in (216,288,360,432):
        k=p//10
        for bins in (2,20,4096):
            scores=torch.randint(-bins,bins,(12,72,p),device='cuda').float()
            own=(torch.arange(72,device='cuda')+p-72)[None,:,None].expand(12,-1,-1)
            ranked=scores.scatter(-1,own,torch.inf)
            ids=ranked.topk(128).indices.int()
            mask=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,ids[...,:k].long(),True)
            ref=ranked.masked_fill(~mask,-torch.inf).topk(k).indices
            y=tie_order(scores,ids,k)
            rows.append(dict(p=p,bins=bins,unequal=int((y!=ref).sum()),tested_ids=y.numel()))
    print(rows,flush=True)
    path=ROOT/'artifacts/video_runtime_v1/selected_tie_order_v1'
    path.mkdir(exist_ok=True)
    (path/'ties.json').write_text(json.dumps(rows,indent=2))
    assert all(r['unequal']==0 for r in rows)
if __name__=='__main__':main()
