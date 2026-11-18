"""Fit slopes on the fixed window, keeping the partial first tail tile in the intercept."""
import csv,json
from pathlib import Path
from statistics import median,mean
root=Path(__file__).resolve().parent
events={k:1000*median(v) for k,v in json.loads((root/'short_events.json').read_text()).items()}
rows=[r for r in csv.DictReader(l for l in (root/'short.csv').read_text().splitlines() if l.startswith('"')) if r['Metric Name']=='gpu__time_duration.sum']
assert len(rows)==11
names=['dense','base']+[f'{c}_{b}' for c in (24,28,32) for b in (3,8,13)]
fixed={key:float(r['Metric Value'])/1000 for key,r in zip(names,rows)}
def fit(x):
    hz=x['dense']/(4*4608*12*128*216*64)
    mma=3456*4096*hz
    page=mean((x[f'32_{b}']-x[f'24_{b}'])/8 for b in (3,8,13))
    block=mean((x[f'{c}_13']-x[f'{c}_3'])/10 for c in (24,28,32))
    offset=mean(x[f'{c}_{b}']-c*page-b*block for c in (24,28,32) for b in (3,8,13))
    err=max(abs(x[f'{c}_{b}']-(offset+page*c+block*b)) for c in (24,28,32) for b in (3,8,13))
    return dict(page_us=page,block_us=block,ref_page_us=192*mma,ref_block_us=102*mma,offset_us=offset,max_residual_us=err,full_control_ratio=x['32_13']/x['base'],ref_offset_us=-32*mma, page_gap_us=32*(page-192*mma), tile_gap_us=13*(block-102*mma), fixed_gap_us=offset+32*mma)
result={key:fit(data) for key,data in [('events',events),('fixed_clock',fixed)]}
print(result)
(root/'short_fit.json').write_text(json.dumps(result,indent=2))
