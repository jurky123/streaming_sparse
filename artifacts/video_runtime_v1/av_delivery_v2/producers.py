"""Trace straight-line AV HMMA operand register definitions in deployed P432 SASS."""
import csv
import gzip
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent
with gzip.open(root.parent/'page_arrival_v1/current.csv.gz','rt') as f:
    next(f)
    rows=list(csv.DictReader(f))
start=int(rows[0]['Address'],16)
hot=json.loads((root.parent/'page_arrival_v1/current.json').read_text())
lo,hi=(int(x,16) for x in hot['exact']['relative_range'])
rows=[r for r in rows if lo<=int(r['Address'],16)-start<=hi]
# Limit reaching definitions to the linear region after the last QK HMMA.
inst=[r['Source'].strip() for r in rows]
hmma=[i for i,s in enumerate(inst) if s.startswith('HMMA.')]
assert len(hmma)==192
defs={}
report=[]
for i in range(hmma[63]+1,len(inst)):
    s=inst[i]
    m=re.match(r'([A-Z][\w.]*) R(\d+),',s)
    if not m:continue
    op,dst=m[1],int(m[2])
    if op.startswith('HMMA.'):
        reg=[int(x) for x in re.findall(r'\bR(\d+)\b',s)]
        assert len(reg)==4
        record={}
        for label,base,width in zip(('p','v','acc'),reg[1:],(4,2,4)):
            hits=[defs.get(n) for n in range(base,base+width)]
            record[label]=[dict(op=inst[j].split()[0],distance=i-j) if j is not None else None for j in hits]
        report.append(dict(pc=hex(int(rows[i]['Address'],16)-start),samples=int(rows[i]['Warp Stall Sampling (Not-issued Samples)']),**record))
    width=4 if op.startswith('HMMA.') or '.128' in op else 2 if '.64' in op else 1
    for n in range(dst,dst+width):defs[n]=i
(root/'producers.json').write_text(json.dumps(report,indent=1))
assert len(report)==128
