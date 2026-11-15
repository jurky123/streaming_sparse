import json,csv,hashlib,gzip,shutil
from pathlib import Path
ROOT=Path('/tmp/e11_set_search');OUT=Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_set_search');OUT.mkdir(exist_ok=True,parents=True)
old={x['layer']:x for x in json.loads(Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_compact/comparison.json').read_text())};rows=[];steps=[];certs=[];manifest=[];costs=[];early=[]
for case in ('horse_field','traffic_crossing'):
 for layer in (1,15,29):
  r=ROOT/case/f'l{layer}';d=json.loads((r/'search_result.json').read_text());risk=d['values']['risk'];ub=d['best_objective']
  if case=='horse_field':
   c=old[layer];lb=c['global_LB'];ub=min(ub,c['incumbent_UB']);status=c['status'];cert_source=c['certificate_source'];runtime=None
  else:
   audit_path=r/'certificate_audit.json';parallel_audit=Path('/tmp/e11_set_search_parallel')/case/f'l{layer}/certificate_audit.json';c=json.loads((audit_path if audit_path.exists() else parallel_audit).read_text());assert c['passed'];lb=c['LB'];ub=min(ub,c['UB']);status=c['status'];cert_source=str(r);runtime=json.loads((r/'branch_result.json').read_text())['runtime']
  parallel=Path('/tmp/e11_set_search_parallel')/case/f'l{layer}'
  if (parallel/'certificate_audit.json').exists():
   pc=json.loads((parallel/'certificate_audit.json').read_text());assert pc['passed'];lb=max(lb,pc['LB']);ub=min(ub,pc['UB']);status='GLOBAL_TOL' if ub-lb<=max(1e-10,1e-6*ub) else 'BOUNDED_INCUMBENT';cert_source=str(parallel);runtime=json.loads((parallel/'branch_result.json').read_text())['runtime']
  assert lb<=ub+1e-12;certs.append(dict(case=case,layer=layer,LB=lb,UB=ub,status=status,runtime=runtime,source=cert_source))
  # Recovery intervals also remain visible for GLOBAL_TOL, reflecting numerical tolerance.
  for method,j in d['values'].items():
   numer=risk-j;denlo=risk-ub;denhi=risk-lb
   if denlo>0:interval=sorted([numer/denhi,numer/denlo]);point=numer/denlo if status=='GLOBAL_TOL' else None
   else:interval=[None,None];point=None
   run=d['exchange_runs'].get(method);row=dict(case=case,layer=layer,head=d['head'],Q=64,P=410,K=41,method=method,nmse=j,risk_gain=1-j/risk,global_LB=lb,global_UB=ub,global_status=status,recovery_point=point,recovery_lower=interval[0],recovery_upper=interval[1],exchange_steps=run['steps'] if run else None,one_swap_status=run['local_certificate']['status'] if run else None,reused=bool(run and run['reused']));rows.append(row)
  for method,trajectory in d['exchange_curves'].items():
   run=d['exchange_runs'][method];costs.append(dict(case=case,layer=layer,method=method,accepted_steps=run['steps'],score_candidates=14760*(run['steps']+1),recorded_wall_seconds=run['wall_seconds'],reused=run['reused'],scope='candidate scores exclude extra direct validation; reused timing is not a matched benchmark'))
   if method=='risk_exchange':
    total=trajectory[0]['nmse']-trajectory[-1]['nmse'];early.append(dict(case=case,layer=layer,first1_fraction=(trajectory[0]['nmse']-trajectory[min(1,len(trajectory)-1)]['nmse'])/total,first2_fraction=(trajectory[0]['nmse']-trajectory[min(2,len(trajectory)-1)]['nmse'])/total,first4_fraction=(trajectory[0]['nmse']-trajectory[min(4,len(trajectory)-1)]['nmse'])/total))
   for t in trajectory:steps.append(dict(case=case,layer=layer,method=method,**t,count=41))
  dst=OUT/case/f'l{layer}';dst.mkdir(exist_ok=True,parents=True)
  for f in r.iterdir():
   if f.is_file() and f.name!='branch_nodes.jsonl' and f.name!='branch_progress.json':shutil.copy2(f,dst/f.name)
  if (r/'branch_nodes.jsonl').exists():
   with (r/'branch_nodes.jsonl').open('rb') as src,gzip.open(dst/'branch_nodes.jsonl.gz','wb') as out:shutil.copyfileobj(src,out)
  if (parallel/'certificate_audit.json').exists():
   target=dst/'parallel';target.mkdir(exist_ok=True)
   for name in ('branch_result.json','branch_frontier.json','certificate_audit.json','soc_point.npy','soc_relaxation.json'):shutil.copy2(parallel/name,target/name)
   with (parallel/'branch_nodes.jsonl').open('rb') as src,gzip.open(target/'branch_nodes.jsonl.gz','wb') as out:shutil.copyfileobj(src,out)
  path=Path(d['input_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==d['input_sha256'];manifest.append(dict(case=case,layer=layer,path=str(path),sha256=d['input_sha256']))
  if case=='traffic_crossing':
   folder=OUT/'inputs';folder.mkdir(exist_ok=True)
   if not (folder/path.name).exists():shutil.copy2(path,folder/path.name)
for name,data in [('comparison',rows),('exchange_steps',steps),('global_bounds',certs),('search_costs',costs),('early_exchange_gain',early)]:
 (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
 with (OUT/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
for name in ('protocol.json','greedy_edge_check.json','parallel_resources.json','endpoint_checks.json'):shutil.copy2(ROOT/name,OUT/name)
code=[]
for folder in ('e11_set_search','e11_compact'):
 for f in Path('/home/shuchenweng/jzk/streaming_sparse/local_experiments',folder).glob('*.py'):code.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(OUT/'manifest.json').write_text(json.dumps(dict(inputs=manifest,code=code,old_horse_certificates='/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_compact',scope='6 head-tiles only, fixed seed101/c5/step0, two prompts, three layers'),indent=2))
lines=['E11 Set-Conditional Search vs Global Optimum','', '范围：178f95a8 / summary_pretrained / G-KVR；Q64/Page64，410页，精确41页含own，R4/rank8，冻结参数。','仅6个固定head–tile：horse_field与traffic_crossing，各seed101/chunk5/step0、layer1/15/29。head=(layer+5)%12，tile=(17*layer+65+101)%72；每实例64Queries。旧3例复用，新加3例，不能视作多seed或多chunk总体结论。','新3例全局搜索各300秒（根节点准备另计）；响应充分利用资源的要求，对3个未闭合实例各追加24 CPU workers×90秒，完整前沿继续，不新增样本。全部最终界逐节点独立回放；未收敛保留LB/UB。','未执行：独立页残差oracle、在线router、下游误差/视频、更多实例扩展。','','结果（Recovery：GLOBAL_TOL按容差证书计算；未认证只报区间）：','case / layer | greedy NMSE | greedy Recovery | exchange NMSE | exchange Recovery | global status']
def recovery(row):
 if row['recovery_point'] is not None:return f"{100*row['recovery_point']:.3f}% (tol)"
 return f"[{100*row['recovery_lower']:.3f}%, {100*row['recovery_upper']:.3f}%]"
for case in ('horse_field','traffic_crossing'):
 for layer in (1,15,29):
  a=next(v for v in rows if v['case']==case and v['layer']==layer and v['method']=='greedy');b=next(v for v in rows if v['case']==case and v['layer']==layer and v['method']=='greedy_exchange');lines.append(f"{case}/L{layer} | {a['nmse']:.12g} | {recovery(a)} | {b['nmse']:.12g} | {recovery(b)} | {b['global_status']}")
lines+=['','交换轨迹：exchange_steps.csv逐步保存step0初始误差、每步NMSE、每步下降delta、相对初始误差和换入/换出页。三条起点分别为Mass、Risk、Greedy，不拼接不同重启轨迹。','构造轨迹：各实例greedy_steps.csv保存从own开始的40次加入；全部15580个加入候选使用原始混合N/Z独立校验。即使所有动作变差也选满41页；这一分支已用合成抵消问题验证。','交换每次遍历全部14760个合法邻居，终止于ONE_SWAP_LOCAL_TOL；旧Mass/Risk轨迹按原输入哈希复用并逐步重算验证。','六个实例中，三种起点的交换最终NMSE相同；纯构造在三个实例略差，交换补齐。旧horse L15简单搜索值略高于此前多起点最好值，legacy_multistart_best仅作为单列历史结果，不当作单条简单搜索。','Recovery区间计算：(Risk-Search)/(Risk-LB) 与 (Risk-Search)/(Risk-UB) 取有序端点，不裁剪负值；不把未认证incumbent当OPT。GLOBAL_TOL仍附完整区间和数值容差。','暂时支持：在这6个实例上，简单交换机制很有效，构造+交换能达到同等结果；还不能据此声称所有实例都接近真实最优，也不能推断可部署的在线收益。所有搜索使用完整原始贡献与64Query，而Mass/Risk保留原M16配置。','','图：exchange_nmse.png；exchange_step_reduction.png；greedy_construction.png。','复现：隔离基线目录 /tmp/group_response_178f95a8 中使用 .venv Python，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1；模块 local_experiments.e11_set_search.search/global_bound/audit，参数 --case <case> --layer <1|15|29>；global_bound附 --seconds 300。']
(OUT/'RESULTS.zh.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:18]))
