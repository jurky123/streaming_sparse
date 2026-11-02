import json,gzip,collections,bisect,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--trace',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
x=json.load(gzip.open(args.trace,'rt'))['traceEvents']
spans=sorted([e for e in x if e.get('cat')=='user_annotation' and e['name'].startswith('sparse/')],key=lambda e:e['ts']);starts=[e['ts'] for e in spans];mapping={}
for e in x:
 if e.get('cat') not in ('cuda_runtime','cuda_driver'):continue
 i=bisect.bisect_right(starts,e['ts'])-1
 if i>=0 and e['ts']+e.get('dur',0)<=spans[i]['ts']+spans[i]['dur']+1:mapping[e.get('args',{}).get('correlation')]=spans[i]['name']
out={}
for e in x:
 if e.get('cat')!='kernel':continue
 name=mapping.get(e.get('args',{}).get('correlation'))
 if not name:continue
 a=out.setdefault(name,{}).setdefault(e['name'],dict(calls=0,total_us=0));a['calls']+=1;a['total_us']+=e['dur']
result={}
for stage,rows in out.items():
 rows=sorted([dict(name=k,**v) for k,v in rows.items()],key=lambda x:-x['total_us']);result[stage]=rows
 print(stage,'total ms',sum(x['total_us'] for x in rows)/1000)
 for row in rows[:12]: print(round(row['total_us']/1000,3),row['calls'],row['name'][:155])
Path(args.out).write_text(json.dumps(result,indent=2))
