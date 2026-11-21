"""Summarize paging controls and the exact softmax-mass diagnostic."""
import json,csv
from pathlib import Path
from statistics import mean
import numpy as np
p=Path(__file__).resolve().parent
def read(f):return json.loads(f.read_text())
methods=['full','ungrouped','natural_matched']
def records(case):return read(p/f'sparse_{case}.json')
r={n:records(n) for n in methods}
lines=['# 自然顺序分页复核','','## 首次滚窗的存活 token 对照','',
       '本组生成 5 秒（80 帧）。第三列变体在首次滚窗前与自然顺序分页相同，滚窗时把部分 chunk 的 48 个存活页换成原分组规则保留的 3,072 个 token，再按原 token 序号分页。它只控制首次滚窗的部分淘汰。','',
       '| 样本 | 原分组 LPIPS | 自然分页 LPIPS | 自然分页、按原规则选存活 token LPIPS |','|---|---:|---:|---:|']
for i,x in enumerate(r['full']):
    lines.append(f"| {x['prompt']} / {x['seed']} | "+' | '.join(f"{r[m][i]['lpips']:.5f}" for m in methods)+' |')
lines.append('| 均值 | '+' | '.join(f"{mean(x['lpips'] for x in r[m]):.5f}" for m in methods)+' |')
overlap=[json.loads(x)['mean_survivor_fraction'] for x in (p/'matched_overlap.jsonl').read_text().splitlines()]
lines+=['',f'四条闭环轨迹上，部分淘汰的 chunk 中原分组与自然分页保留 token 的平均重合率为 {100*mean(overlap[-4:]):.2f}%。','',
        '## 20 条独立视频配对','',
        '5 个场景 × seed 703–706，9 个 chunk。两种方法在同一卡重新生成，以先前冻结的 dense RGB [21,101) 为参照。分段以第 69 帧为首次滚窗边界。','',
        '| 场景 | 原分组 LPIPS | 自然分页 LPIPS | 自然 − 原分组 | 自然分页改善条数 |','|---|---:|---:|---:|---:|']
pairs=[]
for f in sorted((p/'natural_validation/full').glob('*.json')):
    a=read(f);b=read(p/'natural_validation/ungrouped'/f.name)
    pairs.append((f.stem.rsplit('_',1)[0],a,b))
assert len(pairs)==20
for name in sorted({x[0] for x in pairs})+['全部']:
    rows=[(a['lpips'],b['lpips']) for n,a,b in pairs if n==name or name=='全部']
    a,b=np.array(rows).T
    lines.append(f'| {name} | {a.mean():.5f} | {b.mean():.5f} | {(b-a).mean():+.5f} | {(b<a).sum()}/{len(a)} |')
for sl in (slice(0,48),slice(48,80)):
    delta=mean(np.subtract(b['per_frame']['lpips'][sl],a['per_frame']['lpips'][sl]).mean() for _,a,b in pairs)
    lines.append(f"\nRGB [{21+sl.start},{21+sl.stop}) 的平均 LPIPS 差值（自然减原分组）为 {delta:+.6f}。")
for metric in ['ssim','psnr']:
    a,b=[mean(x[i][metric] for x in pairs) for i in (1,2)]
    lines.append(f'\n全窗口平均 {metric.upper()}，原分组 {a:.5f}，自然分页 {b:.5f}。')
lines+=['','## 选页与补偿诊断','',
        '在回填 dense 的四条轨迹上，每个稀疏调用每头轮转取一行 Q，读取实际 LUT，直接用完整 QK softmax 测选中页的概率质量，比较所选页独立输出和含补偿的实际输出。','',
        '| 区间 | 页组织 | 精算页捕获的真实概率质量 | 仅精算页 NMSE | 含补偿 NMSE |','|---|---|---:|---:|---:|']
for sl,label in [(slice(2,6),'首次滚窗前'),(slice(6,7),'首次滚窗后')]:
    for name in ('full','natural'):
        mass,n,e,y,d=np.load(p/f'trace_{name}_mass.npy')[sl].sum(0)
        lines.append(f'| {label} | {name} | {mass/n:.4f} | {e/d:.6f} | {y/d:.6f} |')
lines+=['','## 未精算块补偿的独立复验','',
        '同样的 5 个场景、4 个 seed 和 RGB [21,101) 窗口，两组均在同一张 A100 SXM4 上重新生成。','',
        '| 方案 | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ |','|---|---:|---:|---:|']
cs={}
for name in ('full','no_tail'):
    files=sorted((p/'compensation_validation'/name).glob('*.json'))
    assert len(files)==20
    cs[name]={f.stem:read(f) for f in files}
    means=[mean(r[k] for r in cs[name].values()) for k in ('lpips','ssim','psnr')]
    lines.append(f'| {name} | {means[0]:.5f} | {means[1]:.5f} | {means[2]:.3f} |')
assert cs['full'].keys()==cs['no_tail'].keys()
wins=sum(cs['full'][k]['lpips']<cs['no_tail'][k]['lpips'] for k in cs['full'])
lines+=['',f'开启未精算页补偿的 LPIPS 在 {wins}/20 条轨迹上更低。']
(p/'PAGING.zh.md').write_text('\n'.join(lines)+'\n')
