from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization7'
print('Opt7 热点优化 | 仅 Val；首 chunk / layer0 Dense；独立 query mask')
if (root/'final_summary.json').exists():
 final=json.load(open(root/'final_summary.json'));t=final['all8_timing']
 print(f"最终同卡结果：8 Val 平均 Dense {t['dense_mean_seconds']:.3f}s；近似 Reuse0 {t['fast_mean_seconds']:.3f}s；耗时降低 {t['latency_reduction']*100:.2f}%（基本持平）")
 print('最快版去掉 AV 修正并跨步复用；有个例质量漂移，尚未替换 Opt6 默认入口。')
 print('详细结果：'+str(root/'REPORT.md'))

print('方法                         状态                  生成秒      含 VAE 秒')
for folder in sorted((root/'benchmark').iterdir()):
 if not folder.is_dir():continue
 status=json.load(open(folder/'status.json'))['status']if(folder/'status.json').exists()else'等待'
 records=json.load(open(folder/'timings.json'))['records'][1:]if(folder/'timings.json').exists()else[]
 times='—'if not records else f"{statistics.median(r['generation_seconds']for r in records):.3f}         {statistics.median(r['total_seconds']for r in records):.3f}"
 print(f'{folder.name:28} {status:20} {times}')
print('Oracle 仅诊断性能上限，不是可用推理方案。质量结果见 benchmark_quality.json / quality。')
for f in sorted((root/'quality').glob('rank*.json'))if(root/'quality').exists()else[]:
 r=json.load(open(f));print(f.stem,r.get('status'),r.get('variant',''),r.get('prompt',''))
