"""FP64 first-denominator parity, including equal infinities and large log gaps."""
import json,torch
from pathlib import Path
from .scores_den import first_den
@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(19)
    cases=torch.tensor([-float('inf'),float('inf'),-1000.,-80.,-1.,0.,1.,80.,1000.],device='cuda',dtype=torch.float64)
    lb=cases[:,None].expand(-1,16).contiguous();rx=cases[None,None,:].expand(9,16,-1).contiguous()
    fn=torch.compile(lambda x,y:torch.logaddexp(x[...,None],y),fullgraph=True)
    rows=[]
    for name,x,y in [('edges',lb,rx),('random',torch.randn(864,16,device='cuda',dtype=torch.float64)*80,torch.randn(864,16,16,device='cuda',dtype=torch.float64)*80)]:
        a=fn(x,y);b=first_den(x,y);finite=torch.isfinite(a)&torch.isfinite(b)
        row=dict(case=name,equal=bool(torch.equal(a,b)),max_abs=float((a[finite]-b[finite]).abs().max()),nan_equal=bool(torch.equal(a.isnan(),b.isnan())))
        assert row['equal'] and row['nan_equal'],row
        rows.append(row)
    p=Path('artifacts/video_runtime_v1/den_score_v1/first_den_parity.json');p.write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
