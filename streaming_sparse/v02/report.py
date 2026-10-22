"""Incremental, prompt-first reports with explicit incomplete experiment cells."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
from .model import SparseRouter,RouterConfig

HOME=Path(__file__).resolve().parent


def trace_rows(path):
    text=path.read_text()
    lines=text.splitlines()
    # A live trainer may be in the middle of appending its final line.
    if text and not text.endswith('\n'): lines=lines[:-1]
    return [json.loads(line) for line in lines]


def learning_curves(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=directory/'metrics.jsonl'
    if not path.exists(): return None
    rows=trace_rows(path)
    if not rows: return None
    values=[r['attention_loss'] for r in rows]
    width=min(100,len(rows))
    smooth=[statistics.mean(values[max(0,i-width+1):i+1]) for i in range(len(rows))]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot([r['step']+1 for r in rows],smooth)
    axes[0].set(title='Training event-normalized error (rolling mean)',xlabel='Optimizer update',ylabel='Relative error',yscale='log')
    val=directory/'validation_curve.jsonl'
    if val.exists():
        points=trace_rows(val)
        if points: axes[1].plot([r['step'] for r in points],[r['score'] for r in points],marker='o')
    axes[1].set(title='Validation prompt-macro error',xlabel='Optimizer update',ylabel='Mean of keep 0.3 / 0.2',yscale='log')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    target=directory/'learning_curves.png'
    fig.savefig(target,dpi=150)
    plt.close(fig)
    return target.relative_to(HOME)


def interval(values):
    rng=random.Random(0)
    draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(1000))
    return [draws[25],draws[974]]


def primary(metric):
    return metric.get('macro_prompt_head_relative_error',metric['macro_prompt_relative_error'])


def main():
    runs=HOME/'runs'
    lines=['# Streaming KV Sparse Routing v0.2 研究报告','',
        '报告由已经落盘的结果增量生成，缺失单元格表示尚未完成，不表示零误差。v0.1 原始结果保留，源码定义快照为 `reference_v01.zip`。',
        '本轮主指标先对每个 Prompt 内的各 Layer/Head 分别汇总 SSE / Dense 能量，等权平均 Head 后再跨 Prompt 求均值；同时提供全局 SSE / 能量、Prompt 内能量加权误差与逐 tile Mean/Median/P90/P95。避免高能量 Head 主导主指标，也不能将新的主指标直接与 v0.1 的逐 tile 均值相除。','']
    if (HOME/'status.json').exists():
        state=json.loads((HOME/'status.json').read_text())
        lines+=['当前阶段：'+state['phase']+'；状态：'+state['state']+'。',state['detail'],'']
    p0=runs/'p0/REPORT.md'
    if p0.exists(): lines+=['## P0 诊断','',p0.read_text(),'']
    weights=HOME/'selected_regularizer.json'
    if weights.exists(): lines+=['后续正则配置：`'+weights.read_text().strip()+'`。','']
    lines+=['## 数据覆盖','',
        'Pilot 与 Full 为嵌套划分：Full Train 0–15 / Val 16–19 / Test 20–23；Pilot Train 0–7 / Val 16–17 / Test 20–21。Seed 0/1 始终留在相同 Prompt split。历史 tensor 存于共享 store，view manifest 不复制 KV。','']
    for stage in ('pilot','full'):
        path=HOME/'data'/stage/'coverage.json'
        lines+=['### '+stage,'',path.read_text() if path.exists() else '尚未完成采集。','']
    controls={}
    for name,config in [('transformer',RouterConfig()),('independent',RouterConfig(encoder_type='mlp'))]:
        model=SparseRouter(config)
        controls[name]=dict(encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                            total_parameters=sum(p.numel() for p in model.parameters()))
    lines+=['## 结构与参数控制','',json.dumps(controls,indent=2),'',
        '独立 Block MLP 使用两层 Pre-LN 残差 FFN（256→1536→256），不交换 Block 信息。Frozen 分组复制并冻结完整 KV→group 输入映射，Content 分支仍训练，避免仅冻结 MLP 后位置/ID 映射继续移动分区。Shared A 完全不使用 Layer/Head ID。','']
    lines+=['## 训练与验证曲线','']
    for directory in sorted(runs.glob('*/*/seed*')):
        curve=learning_curves(directory)
        if curve is not None:
            lines.append(f'[{directory.relative_to(runs)} 曲线]({curve.as_posix()})')
    lines.append('')
    complete=defaultdict(list)
    details={}
    for stage in ('pilot','full'):
        lines+=['## '+stage+' 评估','',
            '| Variant | Split | Router Seeds 完成 | Keep | Prompt-macro E | Prompt CI95 | Global E | Tile Mean | Tile P90 | Tile P95 | E / B4 | 实际 Keep |',
            '| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for variant in ('shared','conditioned','spatial','frozen','independent'):
            for split in ('train','val','test'):
                candidates=sorted((runs/stage/variant).glob(f'seed*/eval_{split}/summary.json')) if (runs/stage/variant).exists() else []
                if not candidates:
                    if stage=='full' or variant in ('shared','conditioned'):
                        lines.append(f'| {variant} | {split} | 0 | — | 未完成 | — | — | — | — | — | — | — |')
                    continue
                values=[json.loads(path.read_text()) for path in candidates]
                for keep in (.3,.2,.1):
                    key=f'Ours/keep={keep}'
                    ours=[v['metrics'][key] for v in values]
                    masses=[v['metrics'][f'B4/keep={keep}'] for v in values]
                    # First average training seeds within each prompt, then compute
                    # uncertainty over independent prompts, never over query tiles.
                    prompt_values=defaultdict(list)
                    for v in values:
                        prefix=key+'/prompt_id='
                        for k,m in v['metrics'].items():
                            if k.startswith(prefix): prompt_values[k[len(prefix):]].append(primary(m))
                    per_prompt=[statistics.mean(x) for x in prompt_values.values()]
                    mean=statistics.mean(per_prompt)
                    ci=interval(per_prompt)
                    mass_mean=statistics.mean(primary(x) for x in masses)
                    avg=lambda field:statistics.mean(x[field] for x in ours)
                    lines.append(f"| {variant} | {split} | {len(values)} | {keep} | {mean:.6g} | [{ci[0]:.4g}, {ci[1]:.4g}] | {avg('global_relative_error'):.6g} | {avg('tile_mean'):.6g} | {avg('tile_p90'):.6g} | {avg('tile_p95'):.6g} | {mean/mass_mean:.3f} | {avg('actual_keep'):.5f} |")
                    details[f'{stage}/{variant}/{split}/{keep}']=dict(training_seeds=len(values),prompt_macro=mean,
                        prompt_ci95=ci,mass=mass_mean,delta=mass_mean-mean,ratio=mean/mass_mean,
                        per_prompt=per_prompt,router_seed_values=[primary(x) for x in ours],
                        router_seed_std=statistics.stdev(primary(x) for x in ours) if len(ours)>1 else None)
                for path,v in zip(candidates,values):
                    if v['oracle']:
                        complete[(stage,variant,split)].extend(v['oracle'])
    lines+=['','## Oracle Headroom','',
        '| Stage/Variant/Split | Tiles × Keep | 显著 Headroom 样本 | Mean E_Mass | Mean E_Greedy | Mean Headroom | Recovery 中位数（保留负值） |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,rows in complete.items():
        eligible=[r['recovery'] for r in rows if r['recovery'] is not None]
        mean=lambda field:statistics.mean(r[field] for r in rows)
        recovery=f'{statistics.median(eligible):.4g}' if eligible else '不解释'
        lines.append(f"| {'/'.join(key)} | {len(rows)} | {len(eligible)} | {mean('mass_error'):.6g} | {mean('greedy_error'):.6g} | {mean('oracle_headroom'):.6g} | {recovery} |")
    if not complete: lines.append('| 尚未完成 | — | — | — | — | — | — |')
    lines+=['','Greedy 为精确统计下的启发式参考：前向贪心、Mass 初始化的有限轮增/删/交换，与 Mass 本身取最小误差。它不是数学最优上界。只在 G > max(1e-8, 0.05 E_Mass) 时解释 Recovery，负值保留。每条 Oracle 记录保存实际 token 数；未找到改进不能证明不存在更优组合。',
        'B4 使用同一名义 token 预算；B4_cost_capped 使用 Ours 实际 token 数作为预算上限，仍需核对实际利用率。原始逐 tile 文件保存全部实际成本、Hard Mask 与 query 索引。','',
        '## 研究决策与当前边界','',
        'H1：以 Full 三训练 Seed、独立 Test Prompt 的 Prompt-macro 误差及 B4 比值判断；Pilot 单 Seed 只作稳定性和初步泛化检查。',
        'H2：比较 conditioned / frozen / spatial，并结合 B4 的分区选择能力和 P0 分区单步对照，不能仅用非零梯度作结论。',
        'H3：比较 conditioned Transformer 与参数量相近 independent Block MLP。',
        'H4：优先看 Greedy 绝对改善及预算利用率，再解释 Recovery。',
        'P4 不自动执行。结果接近 Mass 时保持简单 Decoder；主要差距为泛化时研究分布与条件化；训练和验证均差且存在明显 Headroom 时，才逐项考虑 Keep 条件、Query 表示、Query-conditioned Interaction。','',
        '完整配置、随机状态与优化器状态保存在各 run；best.pt 按固定分层验证子集选择，final.pt 为最终更新。验证曲线、训练曲线、梯度分解、Assignment 和 Hard Mask 均保存。最终 Val/Test 使用完整 split。',
        '正式主要对照需要 Full 三 Router Seeds 全部完成；缺失结果时不宣称阶段完成或稳定优于 Mass。Test Prompt 仅 4 个，仍属于初步证据。']
    (HOME/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (HOME/'results.json').write_text(json.dumps(details,indent=2))
    print('Updated',HOME/'REPORT.md',flush=True)


if __name__=='__main__': main()
