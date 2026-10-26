"""Focused degenerate cases for the new fused selector; no model loading."""
import json
from pathlib import Path
import torch
from .fast_route import mask_only
from .opt2.candidates import candidate_sides_fast
from .opt5.select import fused_select

@torch.no_grad()
def main():
    torch.manual_seed(0);torch.set_num_threads(4)
    mask_fn=torch.compile(mask_only,fullgraph=True,dynamic=False)
    cand_fn=torch.compile(candidate_sides_fast,fullgraph=True,dynamic=False)
    cases=[('one_page',17,1),('flat',72,148),('one_peak',128,296),('tied_sides',17,222),('random',72,296),('random512',512,296)]
    rows=[]
    for name,n,b in cases:
        score=torch.zeros((12,n,b),device='cuda')
        if name=='one_peak':score[:,:,-1]=10.
        if name=='tied_sides':score[:,:,:b//2]=1.
        if name.startswith('random'):score.normal_()
        origins=(torch.arange(b,device='cuda')>=b//2).float()
        refmask=mask_fn(score,1.09);refrem,refadd=cand_fn(score,refmask,origins,1.09,128,1,n)
        mask,rem,add=fused_select(score,origins,1.09)
        row=dict(case=name,queries=12*n,pages=b,mask_equal=torch.equal(mask,refmask),remove_equal=torch.equal(rem,refrem),add_equal=torch.equal(add,refadd))
        assert row['mask_equal']and row['remove_equal']and row['add_equal'],row
        rows.append(row);print(row,flush=True)
    out=Path(__file__).resolve().parents[2]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization5/boundary_checks.json'
    out.write_text(json.dumps(dict(status='passed',cases=rows),indent=2)+'\n')

if __name__=='__main__':main()
