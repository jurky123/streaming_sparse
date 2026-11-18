"""Aggregate not-issued PC samples over the two large native loop backedges."""
import csv
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

source=Path(sys.argv[1])
with gzip.open(source,'rt') as f:
    next(f)
    rows=list(csv.DictReader(f))
start=int(rows[0]['Address'],16)
loops=[]
for row in rows:
    branch=re.fullmatch(r'\s*BRA (0x[0-9a-f]+)\s*',row['Source'])
    if branch:
        end=int(row['Address'],16)
        begin=int(branch[1],16)
        if end-begin>4096:
            loops.append((begin,end))
assert len(loops)==2
report={}
for name,(begin,end) in zip(('exact','tail'),loops):
    reasons=Counter()
    ops=Counter()
    hot=[]
    for row in rows:
        pc=int(row['Address'],16)
        if not begin<=pc<=end:
            continue
        reasons.update({k.split()[0]:int(v) for k,v in row.items() if k.endswith('(Not Issued)')})
        samples=int(row['Warp Stall Sampling (Not-issued Samples)'])
        op=row['Source'].split()
        ops[op[int(op[0].startswith('@'))].split('.')[0]]+=samples
        hot.append((samples,hex(pc-start),row['Source'].strip()))
    report[name]=dict(relative_range=[hex(begin-start),hex(end-start)],stalls=dict(reasons.most_common()),consumer=dict(ops.most_common()),hot=sorted(hot,reverse=True)[:10])
source.with_suffix('').with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
