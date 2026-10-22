"""Recompute diagnostic summaries from saved P0 traces, without new GPU work."""
import json
from pathlib import Path
import statistics

HOME=Path(__file__).resolve().parent


def quantile(values,p):
    values=sorted(values)
    index=(len(values)-1)*p
    lo=int(index)
    hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def describe(values):
    return dict(mean=statistics.mean(values),median=statistics.median(values),p90=quantile(values,.9),max=max(values))


def main():
    results={}
    lines=['# P0 补充分解','',
        '本文件仅分析已有诊断轨迹，不重新训练或采集。所有变化率对应单次相邻参数更新，而非跨整个诊断间隔。','',
        '| λ | KL/Attention Loss（加权，中位数） | 正则/Attention 梯度比（中位数/P90/最大值） | 修复比例均值 | 等 token 数分区对照：改善/可比 |',
        '| --- | ---: | --- | ---: | --- |']
    for path in sorted((HOME/'runs/p0').glob('lambda_*/metrics.jsonl')):
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        diag=[r for r in rows if r['cluster_diagnostics']]
        same_cost=[r for r in diag if r['partition_update']['old_kept_tokens']==r['partition_update']['new_kept_tokens']]
        weight=json.loads((path.parent/'config.json').read_text())['collapse_weight']
        result=dict(weight=weight,diagnosed_updates=len(diag),
            attention_loss=describe([r['attention_loss'] for r in rows]),
            weighted_kl_loss_ratio=describe([r['cluster_loss_ratio'] for r in rows]),
            regularizer_gradient_ratio=describe([r.get('weighted_regularizer_gradient_ratio',0) for r in diag]),
            gradient_cosine=describe([r.get('group_gradient_cosine',0) for r in diag]),
            entropy=describe([r['cluster_diagnostics'][0]['soft_assignment_entropy'] for r in diag]),
            max_assignment_probability=describe([r['cluster_diagnostics'][0]['max_assignment_probability_mean'] for r in diag]),
            repair_fraction=describe([r['cluster_diagnostics'][0]['repair_fraction'] for r in diag]),
            hard_soft_mask_gap=describe([r['cluster_diagnostics'][0]['hard_soft_mask_gap'] for r in diag]),
            hard_soft_clustering_gap=describe([r['cluster_diagnostics'][0]['hard_soft_clustering_gap'] for r in diag]),
            matched_cost_partition_updates=len(same_cost),matched_cost_improving_updates=sum(r['partition_update']['partition_only_error_change']<0 for r in same_cost),
            changed_updates=sum(r['partition_update']['permutation_aligned_change']>0 for r in diag))
        results[path.parent.name]=result
        ratio=result['regularizer_gradient_ratio']
        lines.append(f"| {weight:g} | {result['weighted_kl_loss_ratio']['median']:.4g} | {ratio['median']:.4g} / {ratio['p90']:.4g} / {ratio['max']:.4g} | {result['repair_fraction']['mean']:.4g} | {result['matched_cost_improving_updates']}/{len(same_cost)} |")
    lines+=['','梯度比针对 Grouping MLP 参数，实际训练还会更新共享位置/条件映射。中位数不能替代尾部检查；P90 与最大值用于发现局部正则主导。',
        '旧/新分区对照在同一组更新后网络参数下进行。总的改善次数可能伴随不同预算利用率，因此上表额外筛选实际保留 token 数完全相同的更新。样本数很少，只作为训练路径诊断。',
        'Soft/Hard Clustering Gap 使用同一 Hard Block Mask，只改变 Assignment 统计；其 Soft 支持与有效 token 成本不同，不作为最终稀疏模型指标。',
        '完整分布、Soft Assignment 最大概率、entropy、gradient cosine 与两个 Gap 见 analysis.json。']
    target=HOME/'runs/p0'
    (target/'analysis.json').write_text(json.dumps(results,indent=2))
    (target/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__': main()
