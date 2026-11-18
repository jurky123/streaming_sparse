"""Fit slopes on the fixed window, keeping the partial first tail tile in the intercept."""
import csv,json
from pathlib import Path
from statistics import median,mean
root=Path(__file__).resolve().parent
events={k:1000*median(v) for k,v in json.loads((root/'events.json').read_text()).items()}
rows=[r for r in csv.DictReader(l for l in (root/'counters.csv').read_text().splitlines() if l.startswith('"')) if r['Metric Name']=='gpu__time_duration.sum']
assert len(rows)==11
names=['dense','base']+[f'{c}_{b}' for c in (32,48,64) for b in (6,16,26)]
fixed={key:float(r['Metric Value'])/1000 for key,r in zip(names,rows)}
def fit(x):
    hz=x['dense']/(4*4608*12*128*432*64)
    mma=3456*4096*hz
    page=mean((x[f'64_{b}']-x[f'32_{b}'])/32 for b in (6,16,26))
    block=mean((x[f'{c}_26']-x[f'{c}_6'])/20 for c in (32,48,64))
    offset=mean(x[f'{c}_{b}']-c*page-b*block for c in (32,48,64) for b in (6,16,26))
    err=max(abs(x[f'{c}_{b}']-(offset+page*c+block*b)) for c in (32,48,64) for b in (6,16,26))
    return dict(page_us=page,block_us=block,ref_page_us=192*mma,ref_block_us=102*mma,offset_us=offset,max_residual_us=err,full_control_ratio=x['64_26']/x['base'],ref_offset_us=-32*mma)
result={key:fit(data) for key,data in [('events',events),('fixed_clock',fixed)]}
print(result)
(root/'fit.json').write_text(json.dumps(result,indent=2))
