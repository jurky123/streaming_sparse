"""Correlate sampled PCs to the compiler's source markers for each kernel binary."""
import csv
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

folder=Path(__file__).parent
results={}
for name in ('baseline','shared'):
    cubin=folder/(name+'.cubin')
    report=folder/(name+'.ncu-rep')
    disasm=subprocess.check_output(['nvdisasm','-g','-gi',str(cubin)],text=True).split('.text._tail:\n')[-1]
    locations={}
    stage='other'
    for line in disasm.splitlines():
        src=re.search(r'//## File "(.*?)", line (\d+)',line)
        if src:
            file=Path(src[1]).name
            n=int(src[2])
            if file=='gap_exact_round_kernel.py':stage='exact'
            elif file in ('tuned_attention.py','shared.py'):
                lower,upper=(34,66) if name=='baseline' else (28,60)
                stage='tail' if lower<=n<=upper or (n<20 if name=='baseline' else n<14) else 'other'
            else:stage='other'
        addr=re.search(r'/\*([a-f\d]+)\*/',line)
        if addr:locations[int(addr[1],16)]=stage
    source=subprocess.check_output(['ncu','-i',str(report),'--page','source','--print-source','sass','--csv'],text=True)
    rows=csv.DictReader(io.StringIO(source.split('\n',1)[1]))
    agg=defaultdict(Counter)
    base=None
    for row in rows:
        if not row['Address'].startswith('0x'):continue
        pc=int(row['Address'],16)
        if base is None:base=pc
        key=locations[pc-base]
        for metric in ('# Samples','stall_long_sb','stall_short_sb','stall_wait','Instructions Executed'):
            agg[key][metric]+=int(row[metric].replace(',','') or 0)
        op=re.search(r'\b(LDL|STL)\b',row['Source'])
        if op:agg[key][op[1]]+=int(row['Instructions Executed'].replace(',',''))
    results[name]=dict(agg)
    raw=subprocess.check_output(['ncu','-i',str(report),'--page','raw','--csv','--print-units','base'],text=True)
    table=list(csv.reader(io.StringIO(raw)))
    metrics=('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct','sm__warps_active.avg.pct_of_peak_sustained_active')
    results[name]['kernel']={m:table[-1][table[0].index(m)] for m in metrics}
(folder/'counters.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
