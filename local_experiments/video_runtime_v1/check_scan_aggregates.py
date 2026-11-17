"""Direct positive-exclusion checks, including removal of a 1000-log dominant page."""
import json
from pathlib import Path
import torch
from .exclude_scan import exclude_scan
from .scores import exclude_group
@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(43);rows=[]
    for k in (16,32):
        for pos in (-1,0,k//2,k-1):
            logs=torch.randn(4,16,k,device='cuda',dtype=torch.float64)
            values=torch.randn(4,16,k,16,device='cuda',dtype=torch.float64)
            if pos>=0:logs[:,:,pos]+=1000
            ref=exclude_group(logs,values);out=exclude_scan(logs,values)
            errors=[float((x-y).abs().max()) for x,y in zip(ref,out)]
            assert all(torch.isfinite(x).all() for x in out)
            assert torch.allclose(ref[0],out[0],atol=2e-12,rtol=1e-12)
            assert torch.allclose(ref[1],out[1],atol=2e-12,rtol=1e-10)
            rows.append(dict(k=k,dominant_position=pos,log_max_abs=errors[0],output_max_abs=errors[1]))
    p=Path('artifacts/video_runtime_v1/exclude_scan_v1/aggregate_validation.json');p.write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
