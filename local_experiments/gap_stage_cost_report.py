import json,gzip,bisect,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
root=Path(args.root)
watch=json.loads((root/'peer_samples.json').read_text())
assert watch.get('process_scope')=='compute and graphics via NVIDIA XML'
assert watch['samples'] and all(not x.get('peers') and not x.get('error') for x in watch['samples'])
events=json.loads(gzip.decompress((root/'features.trace.json.gz').read_bytes()))['traceEvents']
annotations=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('sparse/') and e.get('cat')=='user_annotation'];threads={}
for e in annotations:threads.setdefault((e['pid'],e['tid']),[]).append(e)
for es in threads.values():es.sort(key=lambda x:x['ts'])
starts={k:[e['ts'] for e in v] for k,v in threads.items()};correlations={}
for e in events:
 if e.get('cat') not in ['cuda_runtime','cuda_driver'] or 'correlation' not in e.get('args',{}):continue
 key=(e['pid'],e['tid']);es=threads.get(key,[]);j=bisect.bisect_right(starts.get(key,[]),e['ts'])-1
 if j>=0 and es[j]['ts']<=e['ts']<=es[j]['ts']+es[j]['dur']:correlations[e['args']['correlation']]=es[j]['name']
groups={}
for e in annotations:groups.setdefault(e['name'],dict(calls=0,kernel_us=0.,kernel_count=0,kernels={} ))['calls']+=1
for e in events:
 if e.get('cat')!='kernel' or e.get('ph')!='X':continue
 name=correlations.get(e.get('args',{}).get('correlation'))
 if name is None:continue
 g=groups[name];g['kernel_us']+=e['dur'];g['kernel_count']+=1
 k=g['kernels'].setdefault(e['name'],dict(calls=0,us=0.,registers_per_thread=e['args'].get('registers per thread'),shared_memory=e['args'].get('shared memory')));k['calls']+=1;k['us']+=e['dur']
assert set(groups)=={'sparse/current_summary','sparse/route','sparse/exact','sparse/compensation','sparse/history_commit'}
for name,g in groups.items():
 assert g['calls']==(29 if name=='sparse/history_commit' else 116), (name,g['calls'])
 assert g['kernel_count']>0
 g['mean_ms']=g['kernel_us']/g['calls']/1000;g['full_calls']=1189 if name=='sparse/history_commit' else 4524;g['extrapolated_seconds']=g['mean_ms']*g['full_calls']/1000
 g['kernels']=sorted([dict(name=n,**v) for n,v in g['kernels'].items()],key=lambda x:-x['us'])
(root/'stage_kernel_costs.json').write_text(json.dumps(groups,indent=2)+'\n')
lines=['# 当前组合的 GPU kernel 成本分解','','通过 CPU sparse 标记范围内的 CUDA runtime 和 driver correlation 关联 GPU kernel。仅计算 kernel 持续时间；不把 profiler 下 CUDA event 包含的排队间隙视为算子计算。','','| 模块 | 平均 kernel 总时长/ms | 全视频次数 | 外推/s |','|---|---:|---:|---:|']
for n,g in groups.items():lines.append(f"| {n} | {g['mean_ms']:.4f} | {g['full_calls']} | {g['extrapolated_seconds']:.4f} |")
lines+=['',f"合计外推 {sum(g['extrapolated_seconds'] for g in groups.values()):.4f} 秒。来自单个稳态 chunk6；早期形状、频率及 profiler 干扰限制其精度。此值不是端到端可消除时间，也不是理论下界。",'', '端到端结果必须引用同版本独立配对测量，不能从本表直接推出加速比。该外推不能证明已经达到理论性能。']
(root/'cost_accounting.zh.md').write_text('\n'.join(lines)+'\n')
for n,g in groups.items():print(n,round(g['mean_ms'],4),round(g['extrapolated_seconds'],4),[(x['name'][:40],round(x['us']/1000,2)) for x in g['kernels'][:3]])
