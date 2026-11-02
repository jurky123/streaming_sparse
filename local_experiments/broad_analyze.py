"""Coverage checks, paired prompt-cluster statistics and plain Markdown report."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .broad_common import ROOT,CHUNKS,cases,guard
from .learned_cache.io import atomic

METHODS=['mean','reconstruction','split']

def run():
    data={};audits=0
    for p in sorted(ROOT.glob('test_worker*/test_*_c*.json')):
        saved=json.loads(p.read_text());assert saved['replay_exact'];audits+=saved['audited_layer_steps']
        for r in saved['rows']:
            for h in range(12):
                key=(r['prompt'],r['seed'],r['chunk'],r['step'],r['layer'],h)
                cell=data.setdefault(key,{})
                assert r['method'] not in cell,(p,key)
                cell[r['method']]={k:r[k][h] for k in ['nmse','sse','energy','max_query_nmse']}
    keys=sorted(data);assert len(keys)==51840,len(keys)
    assert all(set(v)==set(METHODS) for v in data.values())
    expected={(c['name'],c['seed'],ch,s,l,h) for c in cases('test') for ch in CHUNKS for s in range(4) for l in range(30) for h in range(12)}
    assert set(keys)==expected
    arrays={m:{f:np.array([data[k][m][f] for k in keys]) for f in ['nmse','sse','energy','max_query_nmse']} for m in METHODS}
    total={m:dict(nmse=float(v['nmse'].mean()),energy_weighted_nmse=float(v['sse'].sum()/v['energy'].sum()),
        p95_head_event_nmse=float(np.quantile(v['nmse'],.95)),worst_query_nmse=float(v['max_query_nmse'].max())) for m,v in arrays.items()}
    groups={}
    for label,indices in [('prompt',[0]),('seed',[1]),('chunk',[2]),('step',[3]),('layer',[4]),('head',[5]),('layer_head',[4,5])]:
        groups[label]=[]
        for group in sorted({tuple(k[i] for i in indices) for k in keys}):
            mask=np.array([tuple(k[i] for i in indices)==group for k in keys])
            row=dict(group='/'.join(map(str,group)),n=int(mask.sum()))
            for m in METHODS:row[m]=float(arrays[m]['nmse'][mask].mean())
            row['split_vs_reconstruction_pct']=100*(row['split']/row['reconstruction']-1)
            row['split_vs_mean_pct']=100*(row['split']/row['mean']-1)
            groups[label].append(row)
        path=ROOT/f'test_by_{label}.csv'
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(groups[label][0]));w.writeheader();w.writerows(groups[label])
    comparisons={};rng=np.random.default_rng(42);draw=rng.integers(0,6,size=(10000,6))
    for base in ['mean','reconstruction']:
        x=arrays['split']['nmse'];b=arrays[base]['nmse']
        pm=np.array([r['split'] for r in groups['prompt']]);pb=np.array([r[base] for r in groups['prompt']])
        diff=(pm[draw]-pb[draw]).mean(1);relative=(pm[draw].mean(1)/pb[draw].mean(1)-1)*100
        comparisons[base]=dict(delta_nmse=float((x-b).mean()),relative_pct=float((x.mean()/b.mean()-1)*100),
            cluster_bootstrap_delta_ci95=np.quantile(diff,[.025,.975]).tolist(),
            cluster_bootstrap_relative_ci95_pct=np.quantile(relative,[.025,.975]).tolist(),
            head_event_regression_fraction=float((x>b+1e-12).mean()),
            prompt_regressions=int((pm>pb).sum()),layer_regressions=sum(r['split']>r[base] for r in groups['layer']),
            layer_head_regressions=sum(r['split']>r[base] for r in groups['layer_head']))
    selected={m:{} for m in METHODS[1:]};updates={}
    for folder in [ROOT/'train_00_15',ROOT/'train_15_30']:
        status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
        assert all(v==576 for v in status['updates'].values());updates.update(status['updates'])
        saved=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
        for m in selected:
            for layer in saved['models'][m]:
                curve=[(r['scores'][m][layer],r['epoch']) for r in saved['curves']]
                selected[m][layer]=min(curve)[1]
    recovery=json.loads((ROOT/'training_resume_checks.json').read_text());assert recovery['passed']
    for check in ['routing_checks.json','training_ledger_checks.json']:
        assert json.loads((ROOT/check).read_text())['passed']
    summary=dict(coverage=dict(trajectories=24,test_cases=12,test_prompts=6,head_events_per_method=len(keys),queries_per_method=len(keys)*32,
        replay_audited_layer_steps=audits),total=total,comparisons=comparisons,selected_epochs=selected,
        updates=updates,disk=guard(),groups=groups)
    atomic(ROOT/'summary.json',summary)
    lines=['# 全层与独立场景扩大验证','',
        '## 范围与公平性','',
        '冻结 Self-Forcing EMA，原生绝对 RoPE，无 rebase。18 个新 prompt：8 训练、4 验证、6 测试；测试每个 prompt 两个 seed，共 24 条 30 秒对应的 latent 轨迹。全部 30 层 × 12 heads，chunk 5/20/40、4 个去噪步骤、每事件 32 个真实 query。',
        '',f'测试每方法 {len(keys):,} 个 head-events、{len(keys)*32:,} 个 query-head 测量。原始历史 KV 均为 5 chunks（23,040 tokens/head），选中 72 个 64-token 历史页，加当前 4,608 tokens，共 9,216 个计算 tokens/head/query。',
        '', '所有方法共享 response 硬分组，代表严格限制于物理 page 内，每页固定 4 个，不随距离递减。mean 是页内四组硬均值；reconstruction 是重新训练的旧式 mixture 聚合与重建目标；split 是相同聚合器配合历史质量、V 条件误差和边界排序目标。训练相同初始化、相同更新量，每层各自 AdamW，每层每方法 576 updates。每层 best 仅由独立验证集选择，允许 epoch0。',
        '', '这比较的是同一种分组下的聚合/训练目标，不是所有稀疏注意力 baseline 的综合排名，也不是缺失原训练权重的 Opt30 完整复现。',
        '', '## 独立测试结果','',
        '| 方法 | 平均 query NMSE | 总 SSE / 总 target energy | head-event NMSE P95 |',
        '|---|---:|---:|---:|']
    for m in METHODS:
        r=total[m];lines.append(f'| {m} | {r["nmse"]:.8f} | {r["energy_weighted_nmse"]:.8f} | {r["p95_head_event_nmse"]:.8f} |')
    lines+=['','NMSE 对每个 query 的目标均方能量以 1e-5 截断，再等权平均；能量加权指标单独报告。误差来自选中原始 KV 的真实稀疏输出，对照 dense 输出，不是 prototype 输出。','']
    for base,c in comparisons.items():
        ci=c['cluster_bootstrap_relative_ci95_pct']
        lines += [f'- split 相对 {base}：平均 NMSE 变化 {c["relative_pct"]:+.2f}%；按 6 个独立 prompt 聚类 bootstrap 的 95% 区间 [{ci[0]:+.2f}%, {ci[1]:+.2f}%]。退化 prompt {c["prompt_regressions"]}/6、层 {c["layer_regressions"]}/30、layer-head {c["layer_head_regressions"]}/360，head-event 退化比例 {c["head_event_regression_fraction"]:.1%}。']
    lines+=['','区间仅反映这六个测试 prompt 的场景抽样变异，不包含训练初始化不确定性；heads、时间点不作为独立样本。两个测试 seed 在同一 prompt 内合并。','',
        '## 场景与层间差异','','| prompt | mean | reconstruction | split | split 相对 reconstruction |','|---|---:|---:|---:|---:|']
    for r in groups['prompt']:lines.append(f'| {r["group"]} | {r["mean"]:.6f} | {r["reconstruction"]:.6f} | {r["split"]:.6f} | {r["split_vs_reconstruction_pct"]:+.2f}% |')
    lines+=['','完整拆分见 test_by_prompt/seed/chunk/step/layer/head/layer_head.csv；逐事件结果在 test_worker*/，选中 epoch 在 summary.json。','',
        '## 存储、恢复和适用边界','',
        f'- 本轮目录当前占用约 {summary["disk"]["used_mib"]:.1f} MiB，上限 4 GiB，磁盘空闲下限 50 GiB；不保存完整 QKV。',
        '- 原生 pipeline 与紧凑采集前两个 chunk 逐元素一致；clean latent 重建 KV 的抽样 query 通过精确检查。独立测试全部事件再次核对保存的 4 个 query，逐元素一致。',
        '- 实际训练在 4 次更新后中断、恢复到 8 次；layer0 两种目标的权重和 Adam 状态与连续 8 次更新逐元素一致。检查结果见 training_resume_checks.json。',
        '- 每个统计 chunk 原子保存进度，latest/previous 有限轮换、文件锁、输入及源码哈希校验。恢复要重放 clean 前缀；SIGKILL 最多重做尚未提交的工作单元。',
        '- 结果是 dense 轨迹上的离线局部替换，不证明稀疏闭环视频的稳定性或端到端加速。尚需固定方案后做闭环 30 秒视频、预处理摊销与真实 kernel 计时。',
        '- 同原始 KV/选中 KV 上限，尚未实现含摘要及辅助状态的同物理总显存上限。BF16 K/V 原型约为历史原始 KV 的 4/64=6.25%，还需 counts、字典、模型与重排/构建临时空间；当前研究实现用 FP32 聚合，实际占用更大。',
        '- 不依据本测试结果重新挑层或调整超参并重复报告同一 test 为独立结果；后续改法需新的确认集。','']
        # Interpretation is fixed after this completed independent test; no retuning on test.
    c=comparisons['reconstruction'];b=comparisons['mean']
    energy_change=100*(total['split']['energy_weighted_nmse']/total['reconstruction']['energy_weighted_nmse']-1)
    intro=['',f'本轮结论：split 相对重新训练的旧重建目标，平均 NMSE 降低 {-c["relative_pct"]:.2f}%，6/6 个测试场景方向一致；但相对硬均值只降低 {-b["relative_pct"]:.2f}%，置信区间跨零、4/6 场景退化。当前证据支持改进训练目标，但不支持宣称学习聚合稳定胜过四代表硬均值，也不建议据此全层替换。','']
    lines[1:1]=intro
    lines+=['## 观察与下一步优先级','',
        f'能量加权误差的排序不同：split 比 reconstruction 高 {energy_change:.2f}%。不能只选平均 NMSE 这一项宣布全面改善。split 的 head-event P95 也略高于硬均值。',
        '', '相对硬均值，split 在 chunk5 的平均误差增加约 0.25%，chunk20/40 分别降低约 0.27%/0.49%；layer17 的相对退化约 5.44%。层号从0开始。这些是诊断，不能据此在本测试集上选择回退层后仍称独立测试。',
        '', 'split 有20/30层在第6个epoch被验证集选中，说明6epoch预算不等于已经充分收敛；但不能从这个现象推断继续训练必然改善。仅有一组训练初始化，尚未测训练seed敏感性。',
        '', '1. 优先修正训练与实际路由的不一致：当前训练使用实际单query，部署路由使用64-query tile均值。下一轮应以tile为单位选择页面，用tile内多个真实query的输出误差监督，仍保持page内聚合、同选中页数。',
        '2. 将硬均值强基线作为起点：研究以当前response硬分出的四个子组初始化、在page内学习受约束的残差分配；避免近均匀随机初始化时四个代表接近同一个均值。此前全局PCA初始化失败，不等同于这一特定初始化有效；需要单独消融。',
        '3. 补充选择集合相关的V误差目标，而不是只按单页质量/质量排序：一个页的价值取决于已选集合的输出。先离线做一次修正的oracle潜力分析，再判断是否值得增加在线开销。当前V参与原型聚合和loss，但分配器的输入仍只有K，尚未验证V条件化分配。',
        '4. 固定候选后再做新确认prompt、多训练seed、30秒闭环生成和预处理摊销计时；保留硬均值对照。当前离线收益很小，不能用来推断可见视频改善或端到端速度收益。',
        '', '## 补充审计与内存账本','',
        'routing_checks.json：全部30层，分组teacher与直接dense注意力一致；每层一个head/query直接gather 9,216个选中KV得到的输出与page统计一致；page内prototype的counts及加权K/V总量守恒。training_ledger_checks.json：两组完整训练覆盖、有限loss/梯度、验证分数和best选择均核验通过。',
        '', 'memory_accounting.json 给出理论tensor payload：全层原始历史KV 4,050 MiB，含当前KV 4,860 MiB；四代表K/V为253.125 MiB（BF16设想）或506.25 MiB（本轮FP32）；counts约1.978 MiB、query字典5.625 MiB、聚合器参数0.709 MiB。未包含allocator、重排构建临时空间、模型和优化器。不能用这些理论payload当实测峰值显存。','']
    (ROOT/'report.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(total=total,comparisons=comparisons),indent=2))

if __name__=='__main__':run()
