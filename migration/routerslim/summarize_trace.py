"""Reattribute exported Kineto traces, including autograd engine threads.

CUDA kernels are counted once through their runtime launch correlation. Phase
uses the train.backward time interval, since autograd runs on another CPU thread.
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path


PREFIX=('router.','aux.','attention.','backbone.')


def summarize(path):
    events=json.loads(Path(path).read_text())['traceEvents']
    ranges=defaultdict(list);points=defaultdict(list);backward=[];kernels=[]
    for e in events:
        if e.get('ph')!='X':continue
        name=e['name'];cat=e.get('cat','');start=e['ts'];end=start+e.get('dur',0);tid=e['tid']
        if cat=='user_annotation':
            if name=='train.backward':backward.append((start,end))
            if name.startswith(PREFIX)or name=='cache.history_build':ranges[tid].append((start,end,name))
        elif cat=='cpu_op'and 'Sequence number'in e.get('args',{}):points[tid].append(e)
        elif cat=='cuda_runtime'and 'correlation'in e.get('args',{}):points[tid].append(e)
        elif cat in ('kernel','gpu_memcpy','gpu_memset'):kernels.append(e)
    def classify(point_map,range_map):
        result=[]
        for tid,ps in point_map.items():
            rs=sorted(range_map[tid],key=lambda r:(r[0],-r[1]));i=0;stack=[]
            for e in sorted(ps,key=lambda e:e['ts']):
                t=e['ts']
                while i<len(rs)and rs[i][0]<=t:
                    r=rs[i];i+=1
                    while stack and stack[-1][1]<r[0]:stack.pop()
                    stack.append(r)
                while stack and stack[-1][1]<t:stack.pop()
                name=next((r[2]for r in reversed(stack)if r[2].startswith(PREFIX)),None)
                context='history'if any(r[2]=='cache.history_build'for r in stack)else'current'
                phase='backward'if any(a<=t<=b for a,b in backward)else'forward'
                result.append((e,name,context,phase))
        return result
    annotated=classify(points,ranges);forward={};extras=defaultdict(list)
    for e,name,context,phase in annotated:
        if name and e['name'].startswith('aten::')and 'Sequence number'in e['args']:
            forward[e['args']['Sequence number']]=(name,context)
    for e,name,context,phase in annotated:
        if name is None and 'Backward'in e['name']:
            value=forward.get(e['args'].get('Sequence number'))
            if value:extras[e['tid']].append((e['ts'],e['ts']+e['dur'],value[0]))
    for tid,rs in extras.items():ranges[tid].extend(rs)
    launches={}
    for e,name,context,phase in classify(points,ranges):
        if e.get('cat')=='cuda_runtime':launches[e['args']['correlation']]=(phase,context,name or'backbone.other')
    groups=defaultdict(float)
    for e in kernels:
        key=launches.get(e.get('args',{}).get('correlation'),('unknown','unknown','unattributed'))
        groups[key]+=e['dur']/1e6
    total=sum(groups.values())
    rows=[dict(phase=p,context=c,component=n,gpu_seconds=s,gpu_percent=100*s/max(total,1e-30))for(p,c,n),s in sorted(groups.items(),key=lambda x:-x[1])]
    return dict(gpu_kernel_seconds=total,components=rows,
        accounting='Exclusive exported CUDA kernels + memcpy/memset via launch correlation; autograd sequence attribution; checkpoint recomputation included in backward. Kernel sum is not wall time.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args();directory=Path(args.directory)
    report=json.loads((directory/'profile_breakdown.json').read_text())
    for item in report['samples']:
        item['profile']=summarize(directory/f"baseline_history{item['history_tokens']}.trace.json")
        gc.collect()
    (directory/'profile_breakdown.json').write_text(json.dumps(report,indent=2)+'\n')
    import csv
    rows=[dict(history_tokens=item['history_tokens'],**row)for item in report['samples']for row in item['profile']['components']]
    with(directory/'profile_breakdown.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
