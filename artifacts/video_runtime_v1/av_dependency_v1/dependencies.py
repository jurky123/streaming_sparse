"""Join SourceCounters PCs to SM80 control words and in-loop register producers."""
import csv
from collections import Counter
import json
from pathlib import Path
import re
import sys

mode=sys.argv[1]
prefix='/tmp/jzk_dep_'+mode
folder=Path(__file__).resolve().parent
raw={}
for r in csv.reader(open(prefix+'.csv')):
    if r[0]=='File Path': path=Path(r[1]).name
    elif r[0]=='Line No': header=r
    elif r[0].isdigit(): line=int(r[0])
    elif len(r)>3 and r[2].startswith('0x'):
        def metric(name):
            v=r[header.index(name)]
            return int(v) if v.isdigit() else 0
        raw[int(r[2],16)]=(path,line,metric('stall_short_sb'),metric('stall_wait'),r[3].strip())
base=min(raw)
pcstats={addr-base:r for addr,r in raw.items()}
sass=Path(prefix+'.enc').read_text()
pattern=r'/\*([\da-f]+)\*/\s+([^;]+?)\s*;\s*/\* 0x[\da-f]+ \*/\s*/\* 0x([\da-f]+) \*/'
ins={int(pc,16):(op.strip(),int(hi,16)) for pc,op,hi in re.findall(pattern,sass)}
ranges={'page':[(0x41e0,0x6de0),(0x123b0,0x161b0)],'coeff':[(0x4210,0x6e70),(0x120b0,0x15ce0)]}[mode]
records=[]
summary={}
for phase,(start,end) in zip(('exact','compensation'),ranges):
    assert f'BRA {hex(start)}' in ins[end][0]
    outstanding=[[] for _ in range(6)]
    defs={}
    totals=Counter()
    for pc in range(start,end+1,16):
        op,hi=ins[pc]
        name=op.split()[0]
        regs=[int(n) if n!='Z' else -1 for n in re.findall(r'\bR(\d+|Z)\b',op)]
        waitmask=(hi>>52)&63
        dependencies=[]
        for bar in range(6):
            if waitmask & (1<<bar):
                dependencies.extend(outstanding[bar])
                outstanding[bar]=[]
        file,line,short,wait,sampled=pcstats.get(pc,('',0,0,0,''))
        if sampled and name in ('FADD','HMMA.16816.F32.BF16'):
            assert sampled.startswith(name+' '),(hex(pc),op,sampled)
        totals['loop_short']+=short
        roles={}
        if name.startswith('HMMA'):
            for role,reg in zip(('A','B','C'),regs[1:4]):
                if reg>=0 and reg in defs:roles[role]=defs[reg]
        elif name=='FADD':
            roles={f'input{i}':defs[r] for i,r in enumerate(regs[1:]) if r>=0 and r in defs}
        if roles:
            av=(file=='gap_exact_round_kernel.py' and line==153) if phase=='exact' else (file=='tuned_attention.py' and line==18)
            matched=[role for role,prod in roles.items() if prod in dependencies and ins[prod][0].startswith('LDSM')]
            if av and name.startswith('HMMA'):
                totals['av_hmma_short']+=short
                totals['av_hmma_wait']+=wait
                if len(dependencies)==1 and len(matched)==1:
                    totals['sole_LDSM_'+matched[0]+'_short']+=short
                else:
                    totals['other_or_multiple_short']+=short
            records.append(dict(phase=phase,pc=hex(pc),op=op,source=f'{file}:{line}',short=short,wait=wait,
                waitmask=hex(waitmask),barrier_producers=[hex(v) for v in dependencies],
                inputs={r:[hex(p),ins[p][0]] for r,p in roles.items()}))
        for shift in (46,49):
            bar=(hi>>shift)&7
            if bar<6 and pc not in outstanding[bar]:outstanding[bar].append(pc)
        if regs and re.match(r'\S+\s+R\d+[, ]',op) and not name.startswith(('ST','BAR','BRA','@','LDGSTS','DEPBAR')):
            width=(4 if (name.startswith('HMMA') or (name.startswith('LDSM') and name.endswith('.4'))) or '.128' in name else
                   2 if '.64' in name or (name.startswith('LDSM') and name.endswith('.2')) else 1)
            for r in range(regs[0],regs[0]+width):defs[r]=pc
    summary[phase]=dict(totals)
(folder/(mode+'_dependencies.json')).write_text(json.dumps(dict(ranges=ranges,summary=summary,consumers=records),indent=2)+'\n')
print(mode,summary)
