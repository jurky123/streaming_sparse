"""Human-readable status; standard library only, no GPU/model loading."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization9'
def read(p):return json.loads(p.read_text())if p.exists()else{}
def group(label,files):
 rows=[read(p)for p in files];done=sum(r.get('status')=='completed'for r in rows)
 print(f'{label:<16} {done}/{len(files)} 完成')
 for p,d in zip(files,rows):
  if d.get('status')!='completed':print('  '+str(p.parent.relative_to(root))+': '+d.get('status','等待')+' '+d.get('variant','')+(f" 轮次 {d['round']}"if 'round'in d else''))
print('Opt9：候选归约 + K 量化/V 整理融合')
print('最终候选：MemoryFull / MemoryFast；在线 PV 跳过保留为消融。\n')
group('第一阶段视频验证',[root/f'quality/rank{i}.json'for i in range(8)])
group('KV 融合视频验证',[root/f'quality/memory_rank{i}.json'for i in range(8)])
group('第一阶段配对测速',[root/'paired/status.json']+[root/f'paired_val/prompt{i}/status.json'for i in range(1,8)])
group('最终配对测速',[root/'memory_paired/status.json']+[root/f'memory_paired_val/prompt{i}/status.json'for i in range(1,8)])
group('最终模块 profile',[root/f'profile_{v}_step{s}/status.json'for v,s in [('MemoryFull',3),('MemoryFast',3),('MemoryFast',0)]])
f=root/'final_summary.json'
if f.exists():
 d=read(f)['memory_timing'];print('\n8 Val 平均生成时间（每段同卡交替，取中位数）：')
 for name in ['Dense','Opt8Full','MemoryFull','Opt8Fast','MemoryFast']:print(f"  {name:12} {d[name]['generation_seconds']:.3f} s")
 print('\n报告：'+str(root/'REPORT.md'))
else:print('\n最终结果尚未封存；测速与 profile 完成后汇总。')
