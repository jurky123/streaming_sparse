"""Attribute PC samples to the backedges of paired_k.cu, compiled for sm_80."""
import csv,json,re
from collections import Counter,defaultdict
from pathlib import Path
root=Path(__file__).parent
ranges={'exact':(0x6a0,0x3920),'tail':(0x58f0,0x9580)}
sass=Path('/tmp/jzk_full_paired.enc').read_text()
for start,end in ranges.values():
 assert re.search(r'/\*%04x\*/.*BRA 0x%x\s*;'%(end,start),sass)
raw={}
for r in csv.reader(open('/tmp/jzk_paired_own.csv')):
 if not r:continue
 if r[0]=='Line No':header=r
 elif r[0].isdigit():line=int(r[0])
 elif len(r)>3 and r[2].startswith('0x'):
  raw[int(r[2],16)]=(line,r[3],{k:int(r[i].replace(',','') or 0) for i,k in enumerate(header) if k.startswith('stall_') and ' ' not in k})
offset=min(raw)
stats=defaultdict(Counter);at=defaultdict(Counter)
for addr,(line,inst,counts) in raw.items():
 pc=addr-offset
 phase=next((k for k,(s,e) in ranges.items() if s<=pc<=e),'edges')
 stats[phase].update(counts)
 for k,v in counts.items():
  if v:at[(phase,k)][(hex(pc),line,inst.strip())]+=v
summary=dict(phase_samples=stats,top={f'{p}/{k}':[dict(pc=a,line=b,instruction=c,samples=n) for (a,b,c),n in v.most_common(5)] for (p,k),v in at.items()})
(root/'pc_summary.json').write_text(json.dumps(summary,indent=2))
for p,s in stats.items(): print(p,sum(s.values()),s.most_common(6))
