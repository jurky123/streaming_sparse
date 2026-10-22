"""Prompt-paired comparisons across the three independent Router seeds."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import time
from .evaluation_journal import atomic_json

HOME=Path(__file__).resolve().parent


def paired_interval(values):
    rng=random.Random(0)
    samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
    return [samples[250],samples[9749]]


def run(test_only=False):
    protocol=json.loads((HOME/'protocol.json').read_text())
    variants=protocol['variants'];seeds=protocol['router_seeds'];keeps=protocol['evaluation_keep']
    summaries={}
    for variant in variants:
        for seed in seeds:
            for split in (('val','test') if test_only else ('train','val','test')):
                path=HOME/'runs/full'/variant/f'seed{seed}'/f'eval_{split}'/'summary.json'
                summaries[variant,seed,split]=json.loads(path.read_text())
    result=dict(created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),metrics={},paired_comparisons={},oracle={},split_metrics={})
    lines=['# v0.2 Full 实验与机制分析','',
        '24 个 Prompt（Train 16 / Val 4 / Test 4），各 Generation Seeds 0/1；Router Seeds 0/1/2。全部模型冻结 Backbone，仅用 Attention Output Error 训练。',
        '', '主指标先对每个 Prompt 的各 Layer/Head 分别计算全局 SSE / Dense 能量，再等权平均。训练种子标准差单独报告；Prompt 置信区间不把三个训练种子当作额外独立 Prompt。',
        '', '## 完整 Test 主结果','','| Variant | Keep | Ours：三 Seed 均值 ± 标准差 | 同分区 Mass | Ours / Mass | 实际保留率 |','| --- | --- | --- | --- | --- | --- |']
    for variant in variants:
        for keep in keeps:
            rows=[summaries[variant,s,'test']['metrics'] for s in seeds]
            errors=[r[f'Ours/keep={keep}']['macro_prompt_head_relative_error'] for r in rows]
            mass=[r[f'B4/keep={keep}']['macro_prompt_head_relative_error'] for r in rows]
            mean=statistics.mean(errors);baseline=statistics.mean(mass)
            actual=statistics.mean(r[f'Ours/keep={keep}']['actual_keep'] for r in rows)
            result['metrics'][f'{variant}/keep={keep}']=dict(seed_errors=errors,mean=mean,std=statistics.stdev(errors),mass_mean=baseline,
                mass_cost_capped_mean=statistics.mean(r[f'B4_cost_capped/keep={keep}']['macro_prompt_head_relative_error'] for r in rows),ratio=mean/baseline,actual_keep=actual)
            lines.append(f'| {variant} | {keep:.0%} | {mean:.6g} ± {statistics.stdev(errors):.3g} | {baseline:.6g} | {mean/baseline:.3f}× | {actual:.4%} |')
    lines+=['','## Conditioned 的 Train / Val / Test 对照','',
        '每个 Split 内先按独立 Prompt 聚合；不同 Split 的内容不同，差异仅作为分布和拟合诊断。Test-only 阶段报告暂不包含 Train 全量评估。',
        '', '| Split | Keep | Ours：三 Seed 均值 ± 标准差 | 同分区 Mass | Ours / Mass | 全局 SSE / 能量均值 |',
        '| --- | --- | --- | --- | --- | --- |']
    for split in (('val','test') if test_only else ('train','val','test')):
        for keep in keeps:
            rows=[summaries['conditioned',seed,split]['metrics'] for seed in seeds]
            errors=[r[f'Ours/keep={keep}']['macro_prompt_head_relative_error'] for r in rows]
            mass=statistics.mean(r[f'B4/keep={keep}']['macro_prompt_head_relative_error'] for r in rows)
            global_error=statistics.mean(r[f'Ours/keep={keep}']['global_relative_error'] for r in rows)
            mean=statistics.mean(errors)
            result['split_metrics'][f'{split}/keep={keep}']=dict(mean=mean,std=statistics.stdev(errors),
                mass_mean=mass,ratio=mean/mass,global_relative_error_mean=global_error)
            lines.append(f'| {split} | {keep:.0%} | {mean:.6g} ± {statistics.stdev(errors):.3g} | {mass:.6g} | {mean/mass:.3f}× | {global_error:.6g} |')
    lines+=['','## 条件化、分组和 Encoder 的受控比较','',
        'ΔE = E_reference − E_conditioned；正值表示 Conditioned 更好。每个 Test Prompt 内先平均三个 Router Seeds，再对四个独立 Prompt 做配对 bootstrap（10,000 次）。区间跨零表示此规模下证据不充分。',
        '', '| Reference | Keep | ΔE | Prompt-paired 95% CI |','| --- | --- | --- | --- |']
    prompts=json.loads((HOME/'data/full/splits.json').read_text())['test']
    for reference in ('shared','spatial','frozen','independent','same_partition_mass','cost_capped_mass'):
        for keep in keeps:
            deltas=[]
            for prompt in prompts:
                key=f'Ours/keep={keep}/prompt_id={prompt}'
                if reference in ('same_partition_mass','cost_capped_mass'):
                    method='B4' if reference=='same_partition_mass' else 'B4_cost_capped'
                    reference_values=[summaries['conditioned',s,'test']['metrics'][key.replace('Ours/',method+'/')]['macro_prompt_head_relative_error'] for s in seeds]
                else:
                    reference_values=[summaries[reference,s,'test']['metrics'][key]['macro_prompt_head_relative_error'] for s in seeds]
                deltas.append(statistics.mean(reference_values)-statistics.mean(
                    summaries['conditioned',s,'test']['metrics'][key]['macro_prompt_head_relative_error'] for s in seeds))
            ci=paired_interval(deltas)
            result['paired_comparisons'][f'{reference}/keep={keep}']=dict(delta=statistics.mean(deltas),ci95=ci,prompt_deltas=dict(zip(prompts,deltas)))
            lines.append(f'| {reference} | {keep:.0%} | {statistics.mean(deltas):.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] |')
    lines+=['','## 固定 Partition 的 Mass 基线','','| Method | Keep 30% | Keep 20% | Keep 10% |','| --- | --- | --- | --- |']
    for method in ('B0','B1','B2'):
        values=[statistics.mean(summaries['conditioned',s,'test']['metrics'][f'{method}/keep={k}']['macro_prompt_head_relative_error'] for s in seeds) for k in keeps]
        lines.append('| '+method+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    lines+=['','B0=Spatial Mass；B1=Key Clustering Mass；B2=Raw Joint KV Clustering Mass。上面的 spatial Variant 是 B3（固定 Spatial + Learned Router）；B4 是各模型自己的同分区 Mass。']
    lines+=['','## Conditioned 各 Layer/Head 的 Ours / Same-partition Mass','',
        '每格为 Ours / Mass 的绝对相对误差（括号内为倍率），避免 Mass 误差接近零时仅看倍率产生误解。', '',
        '| Layer | Head | Keep 30% | Keep 20% | Keep 10% |','| --- | --- | --- | --- | --- |']
    for layer in protocol['layers']:
        for head in protocol['heads']:
            ratios=[]
            for keep in keeps:
                suffix=f'/keep={keep}/layer_head={layer}:{head}'
                ours=statistics.mean(summaries['conditioned',s,'test']['metrics']['Ours'+suffix]['macro_prompt_head_relative_error'] for s in seeds)
                mass=statistics.mean(summaries['conditioned',s,'test']['metrics']['B4'+suffix]['macro_prompt_head_relative_error'] for s in seeds)
                ratios.append(f'{ours:.4g} / {mass:.4g} ({ours/mass:.3f}×)')
            lines.append(f'| {layer} | {head} | '+' | '.join(ratios)+' |')
    lines+=['','## Conditioned 分 Step、历史长度、生成 Seed 的误差','',
            '每格为误差（三训练 Seed 均值），括号内为同分区 Mass 误差比。',
            '', '| Stratum | Value | Keep 30% | Keep 20% | Keep 10% |','| --- | --- | --- | --- | --- |']
    for field,values in (('step_id',range(4)),('history_tokens',(4680,9360,14040)),('seed',(0,1))):
        for value in values:
            cells=[]
            for keep in keeps:
                suffix=f'/keep={keep}/{field}={value}'
                ours=statistics.mean(summaries['conditioned',s,'test']['metrics']['Ours'+suffix]['macro_prompt_head_relative_error'] for s in seeds)
                mass=statistics.mean(summaries['conditioned',s,'test']['metrics']['B4'+suffix]['macro_prompt_head_relative_error'] for s in seeds)
                cells.append(f'{ours:.5g} ({ours/mass:.2f}×)')
            lines.append(f'| {field} | {value} | '+' | '.join(cells)+' |')
    lines+=['','## Oracle Headroom（仅评估）','','| Split | Keep | Cases | 显著 Headroom Cases | 负 Recovery Cases | 有效 Recovery 中位数 |','| --- | --- | --- | --- | --- | --- |']
    for split in ('val','test'):
        for keep in keeps:
            rows=[r for seed in seeds for r in summaries['conditioned',seed,split]['oracle'] if r['requested_keep']==keep]
            significant=[r for r in rows if r['significant_headroom']]
            recoveries=[r['recovery'] for r in significant]
            entry=dict(cases=len(rows),significant=len(significant),negative_recovery=sum(x<0 for x in recoveries),
                       recovery_median=statistics.median(recoveries) if recoveries else None,
                       mean_mass=statistics.mean(r['mass_error'] for r in rows),mean_greedy=statistics.mean(r['greedy_error'] for r in rows),
                       mean_token_difference=statistics.mean(r['greedy_tokens']-r['mass_tokens'] for r in rows),
                       max_abs_token_difference=max(abs(r['greedy_tokens']-r['mass_tokens']) for r in rows))
            result['oracle'][f'{split}/keep={keep}']=entry
            median=f"{entry['recovery_median']:.3g}" if recoveries else 'N/A'
            lines.append(f"| {split} | {keep:.0%} | {len(rows)} | {len(significant)} | {entry['negative_recovery']} | {median} |")
    lines+=['','## H1–H4 研究问题与下一步','',
        '以下判断只针对本次四个独立 Test Prompts；区间描述当前样本的配对不确定性，不构成大规模分布上的最终结论。']
    def contrast(reference, keep):
        value=result['paired_comparisons'][f'{reference}/keep={keep}']
        lo,hi=value['ci95']
        direction=('Conditioned 误差更低' if lo>0 else
                   'Conditioned 误差更高' if hi<0 else '方向尚不确定')
        return f"Keep {keep:.0%}：{direction}，ΔE={value['delta']:.4g}，95% CI=[{lo:.4g}, {hi:.4g}]"
    lines+=['','### H1：跨 Prompt 与多 Layer/Head 的可学习性','',
        '本实验在三个 Layer、四个 Head 上联合训练，并按 Prompt 隔离测试；不评价训练时完全未见过的 Layer/Head。',
        '',*['- '+contrast('shared',k) for k in keeps],
        '', '同分区 Mass 比值（三训练 Seed 聚合）：'+', '.join(
            f"Keep {k:.0%} = {result['metrics'][f'conditioned/keep={k}']['ratio']:.3f}×" for k in keeps)+'。',
        '与 v0.1 的单 Head 逐 Tile 均值采用不同覆盖和归一化，不能直接以绝对误差大小判断跨轮改善。',
        '', '### H2：Learned Grouping 的作用','',
        'Frozen 保持未训练的 Joint KV Grouping Feature；Spatial 使用固定分组。两项对照应分别解释，不把特征空间变化与端到端学习收益混为一谈。']
    for reference in ('frozen','spatial'):
        lines+=['',reference+':','',*['- '+contrast(reference,k) for k in keeps]]
    lines+=['','### H3：Inter-block Transformer 的作用','',
        'Independent 使用参数量接近的独立 Block MLP；以下 ΔE 为 Independent − Conditioned。',
        '',*['- '+contrast('independent',k) for k in keeps],
        '', '若 ΔE 为负且区间不跨零，当前实验支持独立 Block MLP 更好；不能据此声称 Transformer 的跨 Block 交互已得到验证。',
        '', '### H4：联合选择 Headroom','']
    for keep in keeps:
        entry=result['oracle'][f'test/keep={keep}']
        lines.append(f"- Keep {keep:.0%}：{entry['significant']}/{entry['cases']} 个诊断 Case 存在预设阈值以上的 Greedy 改善；其中负 Recovery {entry['negative_recovery']} 个。Mass/Greedy 平均误差 {entry['mean_mass']:.4g}/{entry['mean_greedy']:.4g}。")
    lines+=['','### 研究决策边界','',
        '先结合以上配对对照判断 Grouping 与 Transformer 的收益，再定位 Decoder。若训练和测试均明显落后于同分区 Mass，且诊断 Tile 存在显著 Greedy Headroom，可在下一轮依次研究 Keep Ratio 条件和 Query Tile 表示，每次只改一个组件。若只有测试落后，优先研究泛化。',
        '本轮不自动增加 Decoder、不使用 Oracle Mask 标签，也不把启发式未找到改善解释为缺乏联合选择空间。成本匹配结果和四 Prompt 的有限覆盖仍约束这些判断。','']
    lines+=['','## 解释范围','',
        '- Shared 与 Conditioned：检验 Layer/Head 条件化。Spatial 与 Conditioned：检验固定分组与学习分组。Frozen 与 Conditioned：检验 Attention Loss 对 Grouping 的训练收益。Independent 与 Conditioned：检验跨 Block Transformer。',
        '- 不同 Variant 各有自己的 Partition，不能只用相同名义 Keep 判断严格成本匹配；应同时查看实际保留率和 B4_cost_capped。',
        '- Greedy 为有限轮启发式参考，不是最优误差上界。仅在 gap > max(1e-8, 5% E_Mass) 时解释 Recovery，负值保留。',
        '- Greedy 与 Mass 使用同一预算上限，但实际 Token 数可能不同；需结合 JSON 中的成本差解释 Headroom。启发式未找到改善不能证明不存在更优联合选择。',
        '- 只有四个独立 Test Prompts，结果属于初步泛化证据。Oracle 抽样是诊断样本，不是全测试分布的无偏收益估计。',
        '- 当前结果不包含高性能 Sparse Kernel 速度或完整视频质量；不把 KV 保留率等同于实际加速。',
        '- Full 评估后半程每卡两项任务并发，评估期间计时包含资源竞争，不作为独占 GPU 性能基准；调度记录与并发设置随包提供。',
        '- Spatial 与自适应分组的块数、边界及尺寸分布可能不同，跨分区对照反映整体策略；Frozen 对照用于单独检查分组特征的学习收益。',
        '- 不自动执行 P4；是否增强 Decoder 依据泛化、Grouping/Encoder 消融及 Oracle Headroom 联合判断。','']
    if test_only:
        lines[0]='# v0.2 完整 Test 机制分析（Train 全量评估仍在运行）'
        lines[1:1]=['','本文件包含全部 15 组模型的完整 Val/Test，但不表示 45 次完整评估和评审包已经完成。']
    (HOME/('reports/TEST_ANALYSIS.md' if test_only else 'reports/FINAL_ANALYSIS.md')).write_text('\n'.join(lines))
    atomic_json(HOME/('reports/test_analysis.json' if test_only else 'reports/final_analysis.json'),result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only',action='store_true')
    run(test_only=parser.parse_args().test_only)
