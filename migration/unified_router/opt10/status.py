from pathlib import Path
import json
root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization10'
def read(p):return json.loads(p.read_text())if p.exists()else{}
def group(label,files):
 rows=[read(p)for p in files];done=sum(r.get('status')=='completed'for r in rows);print(f'{label}: {done}/{len(files)} 完成')
 for p,d in zip(files,rows):
  if d.get('status')!='completed':print('  '+str(p.parent.relative_to(root))+' / '+p.stem+': '+{'loading':'加载模型','warmup':'预热','timing':'测速','evaluating':'质量评估','profiling':'采集 profile','auditing':'检查改变的 query'}.get(d.get('status'),d.get('status','等待'))+' '+d.get('variant','')+(f" 第 {d['round']} 轮"if 'round'in d else''))
print('Opt10：整层独立路由 / FP16 动作头 / 矩统计 / 真实 P 评分')
print('保留逐 query 决策与 AV 修正头；不复用去噪步，不共享 Q64 mask。\n')
for name,f in [('纯执行批量化','full_probe.json'),('矩统计 + 去补零','compact_probe.json')]:
 rows=read(root/f)
 if isinstance(rows,list)and rows and (f!='full_probe.json' or rows[0]['variant']=='Wide'):
  d=rows[0];print(f"微基准 {name}: {d['reference_ms']:.3f} → {d['new_ms']:.3f} ms")
print()
group('8 Val 质量验证',[root/f'quality/rank{i}.json'for i in range(8)])
group('同卡配对测速',[root/'paired/status.json']+[root/f'paired_val/prompt{i}/status.json'for i in range(1,8)])
group('优化后 profile',[root/f'profile_{v}_step3/status.json'for v in ['Opt9','Wide','Head','Moments','Compact']])
group('改变 query 的实际 attention 误差',[root/f'changed_audit/rank{i}.json'for i in range(2)])
probes=['full_probe','tuned_probe','moment_probe','half_probe','quant_probe','fused_probe'];print('整层探针复核: '+str(sum((root/(n+'.json')).exists()for n in probes))+'/6 有结果')
if (root/'REPORT.md').exists()and(root/'final_verification.json').exists():
 d=read(root/'final_summary.json');print('\n最终生成时间（8 Val，各自同卡中位数的均值）：')
 for name,t in d['timing'].items():print(f"  {name:10} {t['generation_seconds']:.3f} s")
 print('\n报告：'+str(root/'REPORT.md'))
else:print('\n尚在验证；微基准不能代替端到端加速。')
