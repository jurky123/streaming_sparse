"""Summarize closed-loop validation of quota-conditioned Pset scopes."""
import csv,json
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parent
def load(path):return json.loads(path.read_text())
sel=load(root/'selection.json')
axes=sel['selected']
def stats(phase,name):
    rows=[load(f) for f in sorted((root/phase/name).glob('*/result.json'))]
    assert len(rows)==(4 if phase=='holdout' else 2),(phase,name)
    return [r['quality']['lpips'] for r in rows],mean(r['generation_seconds'] for r in rows)
rows=[]
lines=['# 条件配额下的集合纠偏范围扫描','',
       '筛选使用 city_walk，seed 701、702，9 个 chunk，生成 105 帧，对同输入且首帧重定位的 dense 评估 RGB [21,101)。保持 intersection_fast 条件配额、首帧保护和 sparse clean。层 0 走 dense，clean 走稀疏注意力且关闭集合纠偏。每个 seed 固定一张卡；跨卡耗时仅作记录。','',
       f"按平均配对 LPIPS 收益从高到低，入选层 {axes['layer']}、头 {axes['head']}、去噪步 {axes['step']}。",'',
       '| 筛选及组合验证 | seed 701 LPIPS | seed 702 LPIPS | 均值 |','|---|---:|---:|---:|']
for phase,name in [('individual','base'),('individual','full')]+[('selected',x) for x in ('old','top_layers','top_heads','top_steps','intersection','additive_quarter')]:
    scores,sec=stats(phase,name)
    lines.append(f"| {name} | {scores[0]:.6f} | {scores[1]:.6f} | {mean(scores):.6f} |")
lines+=['','base 在本报告指保留条件配额、关闭全部集合纠偏。full 指全部稀疏层头和四个去噪步开启纠偏。old 使用原有 21 个位置。','',
        '## 5 秒留出复验','',
        '按整片 80 帧 LPIPS 对同输入 dense，比对同一轨迹上的三个策略。city_walk 为新 seed，short_721 同时更换 prompt 与 seed。','',
        '| 场景 / seed | 关闭 | 旧交集 | 新交集 |','|---|---:|---:|---:|']
for p in ['city_walk','short_721']:
    for seed in (721,722):
        v=[load(root/'holdout'/n/f'{p}_s{seed}/result.json')['quality']['lpips'] for n in ('base','old','intersection')]
        rows.append(v)
        lines.append(f'| {p} / {seed} | '+' | '.join(f'{x:.6f}' for x in v)+' |')
lines.append('| 平均 | '+' | '.join(f'{mean(x):.6f}' for x in zip(*rows))+' |')
(root/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
with (root/'rankings.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['axis','rank','index','LPIPS gain','gain701','gain702'])
    for axis,ranks in sel['rankings'].items():
        for rank,r in enumerate(ranks,1):w.writerow([axis,rank,r['index'],r['gain'],r['seed_gains']['701'],r['seed_gains']['702']])
