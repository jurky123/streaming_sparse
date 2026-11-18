"""Attribute sampled stalls on HMMA consumers inside the two main loops."""
import csv
import gzip
import json
import re
from pathlib import Path

root=Path(__file__).parent
with gzip.open(root/'release_pc.csv.gz','rt') as f:
    next(f)
    rows=[r for r in csv.DictReader(f) if r['Address'].startswith('0x')]
base=int(rows[0]['Address'],16)
pcs={int(r['Address'],16):i for i,r in enumerate(rows)}
metrics=('stall_wait','stall_short_sb','stall_long_sb','Instructions Executed')
def totals(block):
    return {m:sum(int(r[m].replace(',','') or 0) for r in block) for m in metrics}
result={'whole_kernel':totals(rows)}
for i,row in enumerate(rows):
    jump=re.search(r'\bBRA (0x[0-9a-f]+)',row['Source'])
    if not jump:
        continue
    target=int(jump[1],16)
    if target not in pcs or pcs[target]>=i:
        continue
    block=rows[pcs[target]:i+1]
    hmma=[r for r in block if 'HMMA.16816.F32.BF16' in r['Source']]
    if len(hmma) not in (192,102):
        continue
    phase='exact' if len(hmma)==192 else 'compensation'
    result[phase]=dict(first_pc=hex(target-base),last_pc=hex(int(row['Address'],16)-base),
                       loop=totals(block),qk_hmma=totals(hmma[:64]),av_hmma=totals(hmma[64:]))
assert set(result)=={'whole_kernel','exact','compensation'}
(root/'release_pc.json').write_text(json.dumps(result,indent=2))
