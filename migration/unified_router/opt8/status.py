from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization8'
print('Opt8：完整路由计算复用 / FP16 查询投影 / INT8 QK')
print('先优化与验证，后独立计时及 profile；约 15% 保留率，每个 query 独立 mask。')
if (root/'final_summary.json').exists():
 d=json.load(open(root/'final_summary.json'));t=d['all8_timing'];p=d['paired_first_prompt']
 print(f"最终结果：完整路由 {p['Opt7Full']['generation_seconds']:.3f} → {p['QKProj']['generation_seconds']:.3f}s；8 Val 平均 Dense {t['dense_mean_seconds']:.3f}s / QKFast {t['fast_mean_seconds']:.3f}s。")
 print('采用无末层回退的 INT8 QK；量化误差按用户接受口径保留，尚未重训。')
print('\n初步端到端（各 3 次，尚需以下同卡结果）：')
for f in sorted((root/'benchmark').glob('*/timings.json')):
 rows=json.load(open(f))['records'][1:]
 if rows:print(f'{f.parent.name:12s} 生成 {statistics.median(r["generation_seconds"]for r in rows):.3f}s')
print('\n8 段验证：')
for f in sorted((root/'quality').glob('rank*.json')):
 d=json.load(open(f));print(f'{f.stem}: {d.get("status")} {d.get("variant","")} {d.get("prompt","")}')
print('\n同卡计时 / 优化后热点：')
for f in [root/'paired/status.json',*sorted(root.glob('profile_*/status.json')),*sorted(root.glob('paired_val/*/status.json'))]:
 if f.exists():
  d=json.load(open(f));print(f'{f.parent.name}: {d.get("status")} {d.get("variant","")} round={d.get("round","")}')
if(root/'final_summary.json').exists():print('\n最终报告：'+str(root/'REPORT.md'))
