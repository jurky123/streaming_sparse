"""Build durable report from completed, independently audited certificates only."""
import csv,json,hashlib,shutil,gzip
from pathlib import Path
import numpy as np
BASE=Path('/tmp/e11_compact');OUT=Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_compact');OUT.mkdir(parents=True,exist_ok=True)
metrics=json.loads((BASE/'matched_metrics.json').read_text());rows=[];audits=[]
for layer in (1,15,29):
 root=BASE/f'l{layer}';local=json.loads((root/'local_result.json').read_text());ub=local['best_objective'];sources=[]
 for directory in ('/tmp/e11_compact','/tmp/e11_compact_soc','/tmp/e11_compact_secant'):
  r=Path(directory)/f'l{layer}';f=r/'certificate_audit.json'
  if not f.exists():continue
  audit=json.loads(f.read_text());assert audit['passed'];assert abs(audit['UB']-ub)<1e-13,'Regenerate metrics for improved incumbent';sources.append((audit['LB'],directory));audits.append(dict(layer=layer,directory=directory,**audit))
 lb,source=max(sources) if sources else (0.,'no completed independently replayed certificate');gap=ub-lb;status='GLOBAL_TOL' if gap<=max(1e-10,1e-6*ub) else 'BOUNDED_INCUMBENT' if sources else 'AUDIT_PENDING'
 meta=json.loads(Path(f'/tmp/e11_results/contributions/horse_field_s101_c5_l{layer}_s0.json').read_text());mass=local['baseline_mass'];risk=local['baseline_risk'];rows.append(dict(layer=layer,head=meta['head'],tile=meta['tile'],P=410,Q=64,K=41,mass_nmse=mass,risk_nmse=risk,incumbent_UB=ub,global_LB=lb,absolute_gap=gap,relative_gap=gap/ub,status=status,mass_gain_min=1-ub/mass,mass_gain_max=1-lb/mass,risk_gain_min=1-ub/risk,risk_gain_max=1-lb/risk,better_baseline_gain=1-ub/min(mass,risk),certificate_source=source))
 dst=OUT/f'l{layer}';dst.mkdir(exist_ok=True)
 for f in root.iterdir():
  if f.is_file() and (f.suffix in ('.json','.npz') and f.name not in ('compact.npz','lower_coefficients.npz','branch_progress.json','local_progress.json')):shutil.copy2(f,dst/f.name)
 for directory in ('/tmp/e11_compact','/tmp/e11_compact_soc','/tmp/e11_compact_secant'):
  r=Path(directory)/f'l{layer}'
  if not (r/'branch_result.json').exists():continue
  target=dst/Path(directory).name;target.mkdir(exist_ok=True)
  for name in ('branch_result.json','branch_frontier.json','certificate_audit.json','soc_point.npy','soc_relaxation.json'):
   if (r/name).exists():shutil.copy2(r/name,target/name)
  with (r/'branch_nodes.jsonl').open('rb') as src,gzip.open(target/'branch_nodes.jsonl.gz','wb') as dest:shutil.copyfileobj(src,dest)
with (OUT/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'comparison.json').write_text(json.dumps(rows,indent=2));(OUT/'audited_certificates.json').write_text(json.dumps(audits,indent=2))
for name in ('protocol.json','matched_metrics.json','matched_metrics.csv','lower_bound_small_checks.json','node_bound_checks.json','factor_solver_small_checks.json','secant_small_check.json','execution_issues.json','process_memory_snapshot.json','denominator_counterfactual.json','environment.json'):
 if (BASE/name).exists():shutil.copy2(BASE/name,OUT/name)
local_costs=[]
for layer in (1,15,29):
 v=json.loads((BASE/f'l{layer}/local_result.json').read_text());local_costs.append(dict(layer=layer,wall_seconds=v['wall_seconds'],scope='Gram construction + coefficient save + numerical validation +8-start search; excludes initial contribution NPZ read',starts=len(v['runs']),full_one_swap_neighbors=14760))
(OUT/'local_costs.json').write_text(json.dumps(local_costs,indent=2))
stages=[]
for layer in (1,15,29):
 for name in ('miqp_result.json','soc_integer_result.json'):
  f=BASE/f'l{layer}'/name
  if not f.exists():continue
  v=json.loads(f.read_text());stages.append(dict(layer=layer,stage=name,objective_scope='lower-bound model, not original optimum',status=v.get('solver_status',v.get('status')),LB=v.get('original_objective_LB',v.get('LB')),UB=v.get('original_incumbent_UB'),runtime=v['runtime'],cpu_seconds=v['cpu_seconds'],nodes=v['nodes'],solver='SCIP10.0.2',certificate_used_for_original_proof=False))
 for directory in ('/tmp/e11_compact','/tmp/e11_compact_soc','/tmp/e11_compact_secant'):
  f=Path(directory)/f'l{layer}/branch_result.json'
  if not f.exists():continue
  v=json.loads(f.read_text());stages.append(dict(layer=layer,stage=Path(directory).name,objective_scope='original variable-denominator objective',status=v['status'],LB=v['LB'],UB=v['UB'],runtime=v['runtime'],cpu_seconds=v['cpu_seconds'],nodes=v['expanded_nodes'],solver='independent convex-plane B&B',certificate_used_for_original_proof=(f.parent/'certificate_audit.json').exists()))
with (OUT/'solver_stages.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
(OUT/'solver_stages.json').write_text(json.dumps(stages,indent=2))
lines=['E11 compact offline set optimization — Q64 / Page64','', 'Scope: 178f95a8 / summary_pretrained, G-KVR; horse_field/s101/c5/step0, layers1/15/29; all410 pages, exact41 including own; original summaries R4/rank8. No architecture, grouping, router or budget changes.','', 'Real FP64 mixed-attention NMSE (all64 queries, variable denominators):','layer | mass | risk | best feasible UB | audited LB | relative gap | status']
for r in rows:lines.append(f"{r['layer']} | {r['mass_nmse']:.12g} | {r['risk_nmse']:.12g} | {r['incumbent_UB']:.12g} | {r['global_LB']:.12g} | {100*r['relative_gap']:.6g}% | {r['status']}")
lines+=['','相比 risk 的真实最优收益区间：']
for r in rows:lines.append(f"layer{r['layer']}: {100*r['risk_gain_min']:.6f}%–{100*r['risk_gain_max']:.6f}%; 相比 mass: {100*r['mass_gain_min']:.6f}%–{100*r['mass_gain_max']:.6f}%")
lines+=['','结论与范围：','三个实例均找到明显优于 mass/risk 的集合；旧求解器仅返回较好的基线暖启动，本轮真正找到新集合。','D02：risk→最好集合，三个实例的单页残差能量 A 都上升，跨页交叉项 C 都下降，分母 Z/Z_D 都略下降。因此收益主要体现为减少跨页同向误差叠加，而非降低独立页残差能量或有利的分母变化。C 仍为正，不能称为整体净抵消。','这里只证明/约束固定 head–tile 的 pre-W_O 局部目标；不代表在线路由能力、跨head联合最优或视频全局最优。post-W_O 为单head介入、其余head Dense 的测量，见 matched_metrics.csv。','未运行：12/120事件扩展、新的L2/L3视频、M64在线路由信息控制。样本仅3个，不能外推为稳定总体收益。','mass/risk使用原M16在线配置；离线集合搜索读取全部原始页贡献和64个Query。这里的收益同时包含额外信息优势，不能直接宣称可在线实现同等收益，也未排除更强独立风险评分的潜力。','','数值与证书：','所有候选mask共享同一BF16输入和实际存储摘要，FP64页贡献；Gram随机mask和全部一次交换与直接计算一致。最终三方法均通过原FP32 SelectedAttention路径复测，详见 matched_metrics.csv。','交换搜索遍历全部14,760个合法一次交换，mass/risk及6个额外起点；受限二次交换仅用于上界，不参与全局证明。','最终原目标证书只使用独立凸切平面和完整二叉树覆盖；节点分母界对该节点所有可行集合有效。QP、SOC、分母平方割线各模型的最优状态不等于原目标最优。','GLOBAL_TOL 的容差为 max(1e-10,1e-6*UB)，是给定输入及FP64贡献模型的数值证书，不是无限精度证明。','求解器：Gurobi13.0.3受限许可拒绝全规模二次模型；使用SCIP10.0.2/PySCIPOpt6.3.0。原始全局证书不依赖SCIP下界。','数值审计废弃了早期 assumeconvex=True 的直接二次模型界；显式Cholesky平方和模型重新求解并通过小问题枚举核对。最终原目标区间不采用任何被废弃的界。','原SOC整数模型运行/终止状态单列保存，不能将未闭合模型状态写成原目标最优。','本轮核时上限12，整体128；不自动扩展样本。详见 protocol.json。','成本范围：local_costs.json 包含Gram构建、保存、校验和多起点搜索，不含初始NPZ读取；solver_stages.csv 的分支搜索时间不含根节点系数构建和预求解；这些都不是在线视频延迟。','','可复现入口（隔离基线目录 /tmp/group_response_178f95a8）：','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.e11_compact.local --layer 1','同模块 lower / branch_bound / branch_soc / branch_secant；audit_certificate --layer 1 [--root /tmp/e11_compact_soc 或 /tmp/e11_compact_secant]。','模型、输入身份和代码哈希见 manifest.json；选页集合和逐Query D02见各层目录。']
(OUT/'RESULTS.zh.txt').write_text('\n'.join(lines)+'\n')
manifest={'inputs':[],'code':[],'models':[],'invalidated_debug_results':['initial_miqp_debug: assumeconvex direct quadratic bounds; excluded from all final conclusions','unrecognized_convexity_debug: interrupted model with unnecessary binary products','initial_soc_debug: rejected warm starts; superseded'],'raw_snapshots_directory':'/tmp/e11_results/snapshots','output_directory':str(OUT)}
for layer in (1,15,29):
 f=Path(f'/tmp/e11_results/contributions/horse_field_s101_c5_l{layer}_s0.npz');manifest['inputs'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));target=OUT/'inputs';target.mkdir(exist_ok=True)
 if not (target/f.name).exists():shutil.copy2(f,target/f.name)
 for name in ('miqp.cip','soc_integer.cip'):
  f=BASE/f'l{layer}'/name
  if f.exists():
   manifest['models'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
   with f.open('rb') as src,gzip.open(OUT/f'l{layer}'/(name+'.gz'),'wb') as dest:shutil.copyfileobj(src,dest)
for f in Path('/home/shuchenweng/jzk/streaming_sparse/local_experiments/e11_compact').glob('*.py'):manifest['code'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(rows,indent=2))
