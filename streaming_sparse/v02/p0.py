"""Three fixed regularization weights on the existing 4-prompt QKV dataset."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from .train import run
from .data import SnapshotData
from .model import load_router
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='streaming_sparse/data/validation_4p2s')
    p.add_argument('--output',default='streaming_sparse/v02/runs/p0')
    p.add_argument('--steps',type=int,default=100)
    args=p.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    results={}
    for weight in (0.,1e-4,1e-3):
        name=f'lambda_{weight:g}'
        directory=output/name
        config=dict(seed=0,steps=args.steps,lr=1e-4,weight_decay=.01,snapshots_per_step=1,
            tiles_per_event=1,keep_ratios=[.3,.2],collapse_weight=weight,diagnose_every=5,
            validate_every=max(1,args.steps//2),validation_max_snapshots=4,save_every=args.steps,
            log_every=20,router=dict(conditioned=True,grouping='learned',encoder_type='transformer'))
        if not (directory/'final.pt').exists():
            resume=str(directory/'latest.pt') if (directory/'latest.pt').exists() else None
            run(config,args.data,directory,resume=resume)
        model=load_router(directory/'final.pt','cuda')
        summary={}
        for split in ('train','val'):
            path=directory/f'eval_{split}'
            if not (path/'summary.json').exists():
                evaluate(model,SnapshotData(args.data,split),path,baselines=False,oracle_tiles=2 if split=='val' else 0)
            summary[split]=json.loads((path/'summary.json').read_text())
        metrics=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
        diagnosed=[r for r in metrics if r['cluster_diagnostics']]
        info=dict(weight=weight,steps=len(metrics),
            mean_repair_fraction=statistics.mean(r['cluster_diagnostics'][0]['repair_fraction'] for r in diagnosed),
            mean_raw_partition_change=statistics.mean(r['partition_update']['raw_label_change'] for r in diagnosed),
            mean_aligned_partition_change=statistics.mean(r['partition_update']['permutation_aligned_change'] for r in diagnosed),
            partition_improving_updates=sum(r['partition_update']['partition_only_error_change']<0 for r in diagnosed),
            diagnosed_updates=len(diagnosed),
            regularizer_gradient_ratio_median=statistics.median(r.get('weighted_regularizer_gradient_ratio',0) for r in diagnosed),
            gradients_finite=all(all(0<=v<float('inf') for v in r['gradients'].values()) for r in metrics),
            errors={split:{str(k):summary[split]['metrics'][f'Ours/keep={k}'] for k in (.3,.2,.1)} for split in ('train','val')})
        results[name]=info
        (output/'summary.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(info),flush=True)
    lines=['# v0.2 P0 诊断报告','',
        '数据沿用既有 4 Prompt × 2 Seed，不重新采集。三个正则权重使用相同初始化、相同采样顺序和优化配置。',
        '所有数值/预算断言在诊断更新上执行；非诊断更新仍检查一热分组、预算、Loss 与梯度有限性。','',
        '| λ | 修复比例均值 | 对齐后分区变化 | 分区变化改善误差的更新 | 加权正则/Attention 梯度比中位数 | Val 0.3 | Val 0.2 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for v in results.values():
        lines.append(f"| {v['weight']:g} | {v['mean_repair_fraction']:.4f} | {v['mean_aligned_partition_change']:.4f} | {v['partition_improving_updates']}/{v['diagnosed_updates']} | {v['regularizer_gradient_ratio_median']:.4g} | {v['errors']['val']['0.3']['macro_prompt_relative_error']:.6g} | {v['errors']['val']['0.2']['macro_prompt_relative_error']:.6g} |")
    lines+=['','分区变化使用匈牙利匹配消除任意 Cluster ID 置换。分区改善对照在同一更新后网络参数下，分别用旧/新 Hard Partition 重建 H 和 Mask；实际 token 数同时记录，不能仅根据梯度非零宣称分区改善。',
        'Soft/Hard Mask Gap 固定 Hard Partition；Soft/Hard Clustering Gap 固定同一 Hard Block Mask，只替换统计中的 Assignment，属于独立训练诊断。',
        'Greedy 包含前向贪心及 Mass 初始化的有限轮增删交换；均是启发式，不是严格 Oracle 上界。Recovery 仅在 gap > max(1e-8, 5% E_Mass) 时定义，负值保留。',
        'P0 只有一个初始化 Seed 和有限诊断更新，正则对照用于查错和量级判断，不替代三训练 Seed 的正式消融。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
