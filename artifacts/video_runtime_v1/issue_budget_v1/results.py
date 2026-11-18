"""Normalize by issued SM cycles and by the actual 15%-budget matrix work."""
import csv
import json
from pathlib import Path
root=Path(__file__).resolve().parent
result={}
for path in sorted(root.glob('p*.csv')):
    metrics=[{},{}]
    for r in csv.DictReader(l for l in path.read_text().splitlines() if l.startswith('"')):
        i=int(r['ID'])
        assert ('flash_fwd' in r['Kernel Name']) if i==0 else ('exact(' in r['Kernel Name'])
        metrics[i][r['Metric Name']]=float(r['Metric Value'])
    dense,base=metrics
    p=int(path.stem.split('_')[0][1:])
    work=(192*(15*p//100)+102*((p+15)//16-1)-32)*3456*4096
    ratio=work/(4*4608*12*128*p*64)
    row={}
    for key,label in [('gpu__time_duration.sum','elapsed'),('sm__cycles_elapsed.avg','cycles')]:
        row[label+'_fraction']=ratio*dense[key]/base[key]
    for key,label in [('sm__cycles_elapsed.avg.per_second','sm_hz'),('smsp__warps_eligible.avg.per_cycle_active','eligible'),
                      ('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','tensor_pct'),
                      ('smsp__issue_active.avg.pct_of_peak_sustained_active','issue_pct')]:
        row[label]=[m[key] for m in metrics]
    row['sparse_over_dense_instructions_per_flop']=base['smsp__inst_executed.sum']/dense['smsp__inst_executed.sum']/ratio
    result[path.stem]=row
assert len(result)==8
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for k,r in result.items(): print(k,round(r['elapsed_fraction'],4),round(r['cycles_fraction'],4),[round(v/1e6) for v in r['sm_hz']],r['eligible'],r['tensor_pct'],round(r['sparse_over_dense_instructions_per_flop'],3))
