"""Conservative paired RGB quality gate; incomplete results cannot pass."""
import json
from pathlib import Path
from .broad_common import cases
from .learned_cache.io import atomic
ROOT=Path('artifacts/sparse_quality_line_20261031')
def main():
    rows=[];expected=cases('test');labels=['initial','horizon1','horizon2','old16']
    for c in expected:
        folder=ROOT/f'broad_s{c["seed"]}'/c['id'];r=dict(case=c,metrics={})
        for label in labels:
            p=folder/(label+'.json')
            if p.exists():r['metrics'][label]=json.loads(p.read_text())['quality']
        rows.append(r)
    complete=all(len(r['metrics'])==4 for r in rows);gates={}
    for candidate in ['horizon1','horizon2']:
        gates[candidate]={}
        for baseline in ['initial','old16']:
            pairs=[r for r in rows if candidate in r['metrics'] and baseline in r['metrics']]
            bad=[r['case']['id'] for r in pairs if r['metrics'][candidate]['lpips']>r['metrics'][baseline]['lpips']+1e-6 or r['metrics'][candidate]['ssim']<r['metrics'][baseline]['ssim']-1e-6]
            improved=any(r['metrics'][candidate]['lpips']<r['metrics'][baseline]['lpips']-1e-6 for r in pairs)
            gates[candidate][baseline]=dict(status='pending' if not complete else 'pass' if not bad and improved else 'fail',pairs=len(pairs),regressions=bad,mean_lpips_delta=None if not pairs else sum(r['metrics'][candidate]['lpips']-r['metrics'][baseline]['lpips'] for r in pairs)/len(pairs),mean_ssim_delta=None if not pairs else sum(r['metrics'][candidate]['ssim']-r['metrics'][baseline]['ssim'] for r in pairs)/len(pairs))
    atomic(ROOT/'summary.json',dict(complete=complete,expected_units=48,completed_units=sum(len(r['metrics']) for r in rows),gates=gates,rows=rows))
    lines=['# 前段 RGB 质量线验收','','主指标窗口 RGB[21,101)，同模型、同seed，对各自dense参考；6个test场景×2 seeds。保守沿用LPIPS不退化且SSIM不下降的逐场景门槛，必须有LPIPS实际改善。全部单元完成前一律pending。两个seed不视为独立的不同场景。两条训练方法分别对初始化和旧16probe方案报告，不用均值掩盖回退。额外持久预算最多16MiB，逐单元记录实际bytes；两种架构的允许上限有此明确差异。','','| 场景 | seed | 方案 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---:|---|---:|---:|---:|']
    for r in rows:
        for label,q in r['metrics'].items():lines.append(f"| {r['case']['name']} | {r['case']['seed']} | {label} | {q['lpips']:.6f} | {q['ssim']:.6f} | {q['psnr']:.3f} |")
    lines+=['','## 配对门槛','']
    for method,baselines in gates.items():
        for baseline,g in baselines.items():lines.append(f"- {method} 对 {baseline}：{g['status']}，完成 {g['pairs']}/12 对；回退单元：{', '.join(g['regressions']) or '当前未观察到'}。")
    lines+=['','ablation目录的两个既有场景用于定位问题，不作为新增独立测试证据；逐组恢复初始化参数存在交互影响。质量不通过的候选不晋级为质量速度联合通过。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
