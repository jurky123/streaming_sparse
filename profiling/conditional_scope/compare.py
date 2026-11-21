import csv,json
from pathlib import Path
from statistics import mean
p=Path(__file__).resolve().parent/'neighbors'
load=lambda f:json.loads(f.read_text())
frozen=load(p/'frozen.json')
names=['base','old','single_axis','search_mean','search_balanced','top_layers','top_heads','top_steps']
data={n:[load(f) for f in sorted((p/'validation').glob('*/'+n+'.json'))] for n in names}
assert all(len(records)==20 for records in data.values())
def metric(n,k):return [r['quality'][k] for r in data[n]]
def wins(a,b):return sum(x<y for x,y in zip(a,b))
lines=['# 同条件配额下的交集筛选与逐位置替换','',
       '冻结候选后，在 kitchen_cook、forest_run、portrait_turn、reef_fish、tennis_match × seed 703–706 同输入对照。各生成 9 个 chunk，按历史 RGB [21,101) 评分；所有方法同一输入使用同卡重新生成的 dense 参照。质量指标为 20 条等权均值。每条耗时在同样配置的一次预热后测量，混合两张卡，仅作配对参考。','',
       f"邻域搜索共 {frozen['unique']} 个不同交集，每个在 city_walk、seed 701/702 上评分。每个交集包含 21 个可纠偏位置。历史入选规则保留双 seed 均值最低和相对搜索起点两个 seed 均不变差的最优配置。",'',
       '| 冻结方案 | 层 | 头 | 步 | 搜索 LPIPS |','|---|---|---|---|---:|']
for n in ('single_axis','search_mean','search_balanced'):
    r=frozen[n];j=r['policy'];lines.append(f"| {n} | {j['layers']} | {j['heads']} | {j['steps']} | {r['lpips']:.6f} |")
lines+=['','| 跨场景方法 | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | LPIPS 优于旧交集 | 优于单轴新交集 | 秒 |','|---|---:|---:|---:|---:|---:|---:|']
for n in names:
    scores=metric(n,'lpips')
    lines.append(f"| {n} | {mean(scores):.6f} | {mean(metric(n,'ssim')):.6f} | {mean(metric(n,'psnr')):.3f} | {wins(scores,metric('old','lpips'))}/20 | {wins(scores,metric('single_axis','lpips'))}/20 | {mean(r['seconds'] for r in data[n]):.3f} |")
lines+=['','## 各场景平均 LPIPS','',
        '| 场景（各 4 个 seed） | 关闭 | 旧交集 | 单轴新交集 | 两轮单替换 | 替换 − 单轴 |','|---|---:|---:|---:|---:|---:|']
for prompt in sorted({r['prompt'] for r in data['single_axis']}):
    values=[mean(r['quality']['lpips'] for r in data[n] if r['prompt']==prompt) for n in ['base','old','single_axis','search_mean']]
    lines.append(f'| {prompt} | '+' | '.join(f'{v:.6f}' for v in values)+f' | {values[-1]-values[-2]:+.6f} |')
(p/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
with (p/'paired.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['method','prompt','seed','LPIPS','SSIM','PSNR','seconds'])
    for n,records in data.items():
        for r in records:w.writerow([n,r['prompt'],r['seed'],*[r['quality'][k] for k in ('lpips','ssim','psnr')],r['seconds']])
