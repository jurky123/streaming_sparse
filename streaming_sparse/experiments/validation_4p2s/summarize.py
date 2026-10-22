"""Build the held-out experiment report from immutable per-tile results."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    coverage=json.loads((HERE/'coverage.json').read_text())
    results={split:json.loads((HERE/f'eval_{split}/summary.json').read_text())['metrics']
             for split in ('train','val','val_initial')}
    overview={}
    table=[]
    for method in ('B0','B1','B2','B3','Ours','Mass_same_partition','Random'):
        cells=[]
        for keep in (.3,.2,.1):
            key=f'{method}/keep={keep}'
            train=results['train'][key]
            val=results['val'][key]
            overview[key]=dict(train=train,val=val)
            cells.extend((f"{train['mean']:.6g}",f"{val['mean']:.6g}"))
        table.append('| '+method+' | '+' | '.join(cells)+' |')
    text='''# 4 Prompt × 2 Seed 训练 / 验证实验

## 实验设置

使用官方 `MovieGenVideoBench_extended.txt` 的前 4 个不同 Prompt，每个分别使用视频生成 Seed 0、1，形成完整的 8 个组合。前两个 Prompt 的 4 个组合用于训练，后两个 Prompt 的 4 个组合用于验证；两个 split 不共享 Prompt。Seed 指 Backbone 采样 Seed，Router 初始化/训练 Seed 固定为 0。

沿用 layer 4 / head 0（从 0 开始编号）、12 latent frames、全部 4 个 diffusion steps、3 种历史长度（4680 / 9360 / 14040）。每事件采集 4 个至多 32-token 的连续 Q 窗口，离线按 16-token tile 路由，保留完整历史和当前 KV。训练 Ours 与 B3 各 1000 步，训练 Keep=0.3/0.2，学习率 1e-4，配置与 P1 一致。使用固定最终 checkpoint，没有按验证集选择步数或超参数。Keep=0.1 为未参与训练的压力预算。

'''
    text+=f"数据审计：{coverage['events']} events、{coverage['snapshots']} 历史快照；训练 {coverage['train_events']} events，验证 {coverage['val_events']} events。FP32 Dense / backend 最大相对误差为 {coverage['max_dense_backend_relative_error']:.6g}。\n\n"
    text+='''## 完整训练 / 验证对照

以下为同一 split 上逐 Query Tile 相对 Attention Output Error 的均值；数值越低越好。训练和验证各包含 4 个 Prompt/Seed 组合、48 events、336 tiles。各方法按相同名义历史 token 预算评价，实际利用率另行报告。

| 方法 | Train 0.3 | Val 0.3 | Train 0.2 | Val 0.2 | Train 0.1 | Val 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    text+='\n'.join(table)+'\n\n'
    text+='## 结果解读\n\n'
    for keep in (.3,.2):
        train=results['train'][f'Ours/keep={keep}']['mean']
        val=results['val'][f'Ours/keep={keep}']['mean']
        b3=results['val'][f'B3/keep={keep}']['mean']
        mass=results['val'][f'B2/keep={keep}']['mean']
        text+=f'- Keep={keep}：验证集 Ours 相对 B3 的均值误差降低 {(1-val/b3)*100:.1f}%；仍为 B2 Mass 误差的 {val/mass:.2f} 倍；验证/训练误差比为 {val/train:.2f}。\n'
    text+='\n本轮在未见 Prompt 上观察到了学习收益，但还没有达到超越 Mass 的目标。同一学习分区上的 Mass 也优于 Learned Router，说明当前仍有路由选择上的改善空间；不能仅据此判定必须增加 Decoder 复杂度。\n\n'
    text+='## 验证集上的学习效果和实际预算\n\n'
    text+='| Keep | Ours 训练前 | Ours 训练后 | 改善倍数 | 验证 P90 | Ours 实际保留率 | B2 Mass 实际保留率 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for keep in (.3,.2,.1):
        before=results['val_initial'][f'Ours/keep={keep}']['mean']
        ours=results['val'][f'Ours/keep={keep}']
        mass=results['val'][f'B2/keep={keep}']
        text+=f"| {keep} | {before:.6g} | {ours['mean']:.6g} | {before/ours['mean']:.2f} | {ours['p90']:.6g} | {ours['mean_actual_keep']:.6f} | {mass['mean_actual_keep']:.6f} |\n"
    text+='\n## 每个验证 Prompt / Seed 的 Ours 结果\n\n'
    text+='| Prompt 索引 | 生成 Seed | Keep=0.3 | Keep=0.2 | Keep=0.1 |\n| --- | --- | ---: | ---: | ---: |\n'
    for prompt in protocol['prompts']:
        if prompt['split']!='val':
            continue
        for seed in (0,1):
            vals=[results['val'][f"Ours/keep={keep}/prompt_seed={prompt['prompt_id']}:{seed}"]['mean']
                  for keep in (.3,.2,.1)]
            text+=f"| {prompt['index']} | {seed} | "+' | '.join(f'{x:.6g}' for x in vals)+' |\n'
    text+='\n## 同分区、实际 token 数相同的配对比较\n\n'
    text+='| Keep | 完全相同 token 数的 tile 数 | Mass Error − Ours Error |\n| --- | ---: | ---: |\n'
    for keep in (.3,.2,.1):
        m=results['val'][f'Mass_same_partition/keep={keep}']
        delta=m['exact_token_matched_delta']
        text+=f"| {keep} | {m['exact_token_match_count']} | {delta:.6g} |\n" if delta is not None else f"| {keep} | 0 | 无匹配样本 |\n"
    text+='''
正的配对差值表示 Ours 误差更低，负值表示 Mass 更低。该筛选只是一部分 tiles 的诊断，必须结合匹配数量看待。完整结果包含每个 tile 的实际 token 数、同分区预算差值、Soft Error 与 Hard/Soft Gap。

## 运行与复现

- `protocol.json`：Prompt 原文、索引、ID、固定 split、生成和训练 Seed、采集命令。
- `coverage.json`：8 个 Prompt/Seed 组合的完整覆盖和数值检查。
- `ours_config.json`、`b3_config.json`：训练参数。
- `ours/`、`b3/`：初始化权重、最终权重、可恢复训练状态、逐步梯度和聚类统计。
- `eval_train/`、`eval_val/`、`eval_val_initial/`：原始逐 tile 结果和分层汇总；含 Prompt、Seed、diffusion step 和历史长度。
- `commands.jsonl` 与 `*.log`：实际命令与运行日志。
- `../../data/validation_4p2s/`：原始 QKV 和 split。

这是两个未见过 Prompt、各两个视频 Seed 的小规模验证；只有两个独立验证 Prompt，不能据此声称大规模泛化或统计稳定性。没有在本实验中测量视频质量或 Sparse Kernel 速度。报告的置信区间按 Prompt 重采样，不能将生成 Seed 当作新增的独立 Prompt。
'''
    (HERE/'REPORT.md').write_text(text)
    (HERE/'results.json').write_text(json.dumps(overview,indent=2))
    print(text)


if __name__=='__main__':
    main()
