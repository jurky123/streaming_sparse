import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = {}
for p in (216, 288, 360, 432):
    with (root/f'dense_{p}.csv').open() as f:
        rows = csv.DictReader(line for line in f if line.startswith('"'))
        cycles = [float(row['Metric Value']) for row in rows if row['Metric Name']=='sm__cycles_elapsed.avg']
    dense, base, early = cycles
    mma = 192*(15*p//100) + 102*((p+15)//16-1) - 32
    ratio = mma*3456*4096 / (4*4608*12*128*p*64)
    summary[p] = dict(base=ratio*dense/base, early=ratio*dense/early)
(root/'dense.json').write_text(json.dumps(summary, indent=2)+'\n')
print(summary)
