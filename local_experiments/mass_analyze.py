"""Independent confirmation, no test-based method or layer selection."""
import csv,json
import numpy as np
import torch
from .mass_common import ROOT,cases,CHUNKS,guard
from .mass_router import METHODS
from .learned_cache.io import atomic

def run():
    methods=['mean']+METHODS;data={};audits=0
    for p in sorted(ROOT.glob('test_worker*/test_*_c*.json')):
        obj=json.loads(p.read_text());assert obj['replay_exact'];audits+=obj['audited_layer_steps']
        for r in obj['rows']:
            for h in range(12):
                key=(r['prompt'],r['seed'],r['chunk'],r['step'],r['layer'],h)
                cell=data.setdefault(key,{})
                assert r['method'] not in cell
                cell[r['method']]={f:r[f][h] for f in ['nmse','sse','energy','max_query_nmse','retained_mass']}
    keys=sorted(data)
    expected={(c['name'],c['seed'],ch,s,l,h) for c in cases('test') for ch in CHUNKS for s in range(4) for l in range(30) for h in range(12)}
    assert set(keys)==expected and len(keys)==51840
    assert all(set(v)==set(methods) for v in data.values())
    a={m:{f:np.array([data[k][m][f] for k in keys]) for f in ['nmse','sse','energy','max_query_nmse','retained_mass']} for m in methods}
    total={m:dict(nmse=float(v['nmse'].mean()),energy_nmse=float(v['sse'].sum()/v['energy'].sum()),p95=float(np.quantile(v['nmse'],.95)),worst_query=float(v['max_query_nmse'].max()),retained_mass=float(v['retained_mass'].mean())) for m,v in a.items()}
    groups={}
    for name,cols in [('prompt',[0]),('seed',[1]),('chunk',[2]),('step',[3]),('layer',[4]),('head',[5]),('layer_head',[4,5])]:
        groups[name]=[]
        for g in sorted({tuple(k[i] for i in cols) for k in keys}):
            mask=np.array([tuple(k[i] for i in cols)==g for k in keys]);row=dict(group='/'.join(map(str,g)),n=int(mask.sum()))
            row.update({m:float(a[m]['nmse'][mask].mean()) for m in methods});groups[name].append(row)
        with (ROOT/f'test_by_{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(groups[name][0]));w.writeheader();w.writerows(groups[name])
    pairs=[(m,'mean') for m in METHODS]+[('mass_mean','control'),('mass_var','control'),('mass_var','mass_mean')]
    rng=np.random.default_rng(42);draw=rng.integers(0,6,(10000,6));comparisons={}
    for m,b in pairs:
        pm=np.array([r[m] for r in groups['prompt']]);pb=np.array([r[b] for r in groups['prompt']])
        samples=100*(pm[draw].mean(1)/pb[draw].mean(1)-1)
        comparisons[f'{m}_vs_{b}']=dict(relative_pct=100*(total[m]['nmse']/total[b]['nmse']-1),ci95_pct=np.quantile(samples,[.025,.975]).tolist(),
            prompt_regressions=int((pm>pb).sum()),layer_regressions=sum(r[m]>r[b] for r in groups['layer']),layer_head_regressions=sum(r[m]>r[b] for r in groups['layer_head']),
            head_event_regressions=float((a[m]['nmse']>a[b]['nmse']+1e-12).mean()),energy_change_pct=100*(total[m]['energy_nmse']/total[b]['energy_nmse']-1))
    epochs={m:{} for m in METHODS}
    for folder in [ROOT/'train_00_30']:
        st=json.loads((folder/'status.json').read_text());assert st['status']=='completed' and all(v==576 for v in st['updates'].values())
        saved=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
        for m in METHODS:
            for l in saved['models'][m]:epochs[m][l]=min((c['scores'][m][l],c['epoch']) for c in saved['curves'])[1]
    for name in ['router_checks.json','training_resume_checks.json','training_ledger_checks.json','initial_baseline_checks.json']:
        assert json.loads((ROOT/name).read_text())['passed']
    summary=dict(total=total,comparisons=comparisons,groups=groups,selected_epochs=epochs,
        coverage=dict(head_events=51840,query_head_measurements=1658880,replay_layer_steps=audits),disk=guard())
    atomic(ROOT/'summary.json',summary)
    alpha=torch.cat([saved['models']['mass_var'][l]['alpha'] for l in range(30)]).numpy()
    summary['alpha']=dict(min=float(alpha.min()),median=float(np.median(alpha)),max=float(alpha.max()),positive=int((alpha>1e-6).sum()),values=alpha.tolist())
    oldroot=ROOT.parent/'tile_anchor_20261030';control_delta=0.
    for part in ['train_00_15','train_15_30']:
        previous=torch.load(oldroot/part/'best.pt',weights_only=False,map_location='cpu')['models']['anchor_tile']
        for l,state in previous.items():
            for k,v in state.items():control_delta=max(control_delta,(v-saved['models']['control'][l][k]).abs().max().item())
    summary['control_previous_best_max_abs']=control_delta
    atomic(ROOT/'summary.json',summary)
    lines=['# 共享页质量蒸馏与query方差：全层独立确认','',
        '本轮独立确认集是新6个prompt×2seed(606/707)，不复用前两轮测试集。训练8prompt、验证4prompt沿用，teacher字典仅来自训练。冻结Self-Forcing EMA、绝对RoPE，无rebase。30层×12heads，chunk5/20/40、四步去噪，每方法51,840个head-events、1,658,880个query-head测量。',
        '', '同5chunk历史23,040token/head，选择72个64-token历史页，当前4,608token全保留，共9,216token/head/query；每页4代表，不跨page、不随距离递减。',
        '', 'control为上一轮硬均值起点聚合器+tile版split目标；mass_mean用同一聚合器，改为KL蒸馏共享页质量；mass_var在mass_mean上增加query对角方差项。三组初始评分均等于硬均值。每层每方法6epoch/576次更新、AdamW lr.001、clip1；验证集按真实稀疏输出NMSE选择各层best，允许epoch0。不按确认集选择层或调参。',
        '', '| 方法 | 平均 query NMSE ↓ | 能量加权误差 ↓ | head-event P95 ↓ | 保留真实注意力质量 ↑ |','|---|---:|---:|---:|---:|']
    for m in methods:
        r=total[m];lines.append(f'| {m} | {r["nmse"]:.8f} | {r["energy_nmse"]:.8f} | {r["p95"]:.8f} | {r["retained_mass"]:.6f} |')
    lines+=['', '所有误差来自所选原始KV的真实输出，不是prototype输出。平均NMSE逐query以目标均方能量（下限1e-5）归一化；能量加权误差为总SSE/总目标能量。保留质量高不一定输出误差低。',
        '', '## 成对比较','', '| 对比 | NMSE变化 | prompt聚类95%区间 | 退化prompt | 退化层 | 能量加权变化 |','|---|---:|---:|---:|---:|---:|']
    for key,c in comparisons.items():
        lo,hi=c['ci95_pct'];lines.append(f'| {key} | {c["relative_pct"]:+.3f}% | [{lo:+.3f}%, {hi:+.3f}%] | {c["prompt_regressions"]}/6 | {c["layer_regressions"]}/30 | {c["energy_change_pct"]:+.3f}% |')
    lines+=['', '负值表示误差下降；bootstrap以6个prompt为簇，两seed在场景内合并，10,000次重采样、seed42。区间未校正多重比较，仅作为探索性不确定性描述；训练只有一个初始化。',
        '', '## 学习目标与方差公式','',
        '每个训练事件抽8个tile，每tile4个真实query。teacher先计算每个query在完整历史+当前KV上的归一化attention，对四个query的页质量求平均，再将历史部分归一化为分布t。student历史页分数的softmax为p，loss=KL(t||p)。先只对历史归一化再平均是另一目标，本实验没有这样做。当前KV通过完整attention归一化影响teacher权重，实际始终dense保留。',
        '', 'mass_mean只用tile均值μ；mass_var用全部64个query的逐维总体方差σ²，每个prototype的评分为 μ·K/√128 + log16 + α·(σ²·K²)/(2×128)，四prototype再logsumexp到page。α为每层每head一参数，初始化0、每步投影[0,2]。忽略非对角协方差，是高斯矩近似启发，非归一化attention期望的精确公式。',
        '', '聚合器仍在每个page已有的四个16-token硬子组内做有界可学习加权K/V，固定count16；不是token质量守恒的软分区。mass两组只优化质量KL，没有额外V/边界排序loss，所以与control的比较同时改变监督目标及loss组合。三组loss尺度不同仍采用相同学习率/clip，不能当作各自充分调参后的上限比较。',
        '', '验证集选择的epoch0层数：'+ '，'.join(f'{m}: {sum(v==0 for v in epochs[m].values())}/30' for m in METHODS)+'。epoch0为硬均值；不能将整体改善理解为每层均需学习。',
        '', f'选中mass_var模型的α：min={summary["alpha"]["min"]:.6f}，median={summary["alpha"]["median"]:.6f}，max={summary["alpha"]["max"]:.6f}，非零{summary["alpha"]["positive"]}/360。逐head数值在summary.json。',
        '', f'control与上一轮anchor_tile选中权重的最大绝对差为{control_delta:.9g}（0表示逐元素一致）。',
        '', '## 场景拆分','', '| 场景 | mean | control | mass_mean | mass_var |','|---|---:|---:|---:|---:|']
    for r in groups['prompt']:lines.append('| '+r['group']+' | '+' | '.join(f'{r[m]:.6f}' for m in methods)+' |')
    lines+=['', '完整prompt/seed/chunk/step/layer/head/layer_head拆分见test_by_*.csv；逐事件结果在test_worker*/。',
        '', '## 工程验证与成本边界','',
        '- 初始全验证集三组与硬均值的51,840次head-event对比完全一致，见initial_baseline_checks.json。',
        '- 真实重放检查覆盖teacher归一化顺序、control评分与旧实现一致、α=0时方差版与均值版严格一致、方差项确实改变评分、有限非零梯度。',
        '- 实际训练4次更新后停止、恢复至8次；layer0三组权重和Adam状态与连续训练逐元素一致。每层独立优化器，原子chunk提交、滚动checkpoint、文件锁、身份哈希，恢复需重建clean前缀。',
        '- 全训练记录覆盖、有限loss/梯度、验证分数及best选择审计通过。确认测试4,320个layer-step事件中的抽样query与原采集逐元素一致。',
        f'- 新目录当前占用约{summary["disk"]["run_mib"]:.1f} MiB，4GiB预算、磁盘空闲50GiB下限。旧train/val/字典经只读使用的symlink复用，不复制全QKV。',
        '- 三组同原始/选中KV上限，尚非同物理总显存部署。四个K/V代表的全层FP32 payload约506.25MiB（BF16设想253.125MiB）；缓存FP32 K²另加253.125MiB。研究评分实现每次计算K²，不持久缓存，因此付出额外计算和临时空间。α仅360个FP32参数（1,440bytes）。',
        '- 方差版增加query方差归约及一个Qvar×K²矩阵乘法。路由microbenchmark见routing_benchmark.json；其耗时不包含分组/稀疏kernel/整模型，不能代替端到端加速验证。',
        '- 这是dense轨迹上的离线替换，未验证闭环30秒视频稳定性。上一轮约30%的潜力来自精确信息诊断，不是本轮应达到或已达到的可部署收益。','']
    bench=json.loads((ROOT/'routing_benchmark.json').read_text())
    summary['routing_benchmark']=bench
    atomic(ROOT/'summary.json',summary)
    b=bench['median_across_layers_ms']
    lines+=['## 路由算子计时','',
        '使用验证集真实Q/K/V、选中的mass_var权重，在6个固定层(0/4/14/17/24/29)测12heads×72个query tile的完整历史页打分与top72。所有打分对照共用相同prototype，mean对照只关闭方差项；包括query均值/方差归约。每项预热15次，再7组×20次CUDA event计时，报告各层中位数的中位数。',
        '', '| FP32 PyTorch eager算子 | 毫秒 |','|---|---:|',
        f'| 均值路由 | {b["mean"]:.5f} |',f'| 方差路由，每次计算K² | {b["variance_dynamic_k2"]:.5f} |',
        f'| 方差路由，缓存K²及log-count | {b["variance_cached_k2"]:.5f} |',
        f'| 一个新chunk的加权聚合（不含分组） | {b["incoming_chunk_aggregation_only"]:.5f} |',
        '', f'动态K²的方差路由耗时约为均值路由的{bench["dynamic_ratio"]:.2f}倍，缓存K²版本约{bench["cached_ratio"]:.2f}倍。缓存版本同时复用log-count，其top72选择与动态版本逐元素一致。',
        '', '这是共享机器上的局部FP32 eager实现，不是原仓库优化kernel，也未包含实际稀疏attention/全模型/视频解码；不能从该倍率推出整体变慢或加速。缓存K²将计算成本换成额外摘要存储，端到端收益必须另行验证。','']
    mean_c=comparisons['mass_mean_vs_mean'];var_c=comparisons['mass_var_vs_mean'];extra=comparisons['mass_var_vs_mass_mean']
    lo,hi=var_c['ci95_pct']
    intro=['',
        f'本轮结果：共享页质量蒸馏相对硬均值平均NMSE降低{-mean_c["relative_pct"]:.2f}%；加query方差后降低{-var_c["relative_pct"]:.2f}%（prompt聚类变化区间[{lo:.2f}%, {hi:.2f}%]）。两种蒸馏均在6个新确认场景改善。方差相对均值蒸馏额外降低{-extra["relative_pct"]:.2f}%，区间也未跨零。',
        '', f'方差版相对硬均值的能量加权误差降低{-var_c["energy_change_pct"]:.2f}%；但仍有{var_c["layer_regressions"]}/30层、{var_c["layer_head_regressions"]}/360个layer-head和{var_c["head_event_regressions"]:.1%}的head-events退化。因此只能说整体及场景平均得到改善，不能说所有位置都改善。',
        '', f'在线代价：均值路由约{b["mean"]:.3f}ms，方差动态K²约{b["variance_dynamic_k2"]:.3f}ms，缓存K²及log-count约{b["variance_cached_k2"]:.3f}ms（12heads×72tiles的FP32 eager路由）。这些不是整模型耗时；尚未验证闭环视频或端到端加速。','']
    lines[1:1]=intro
    lines+=['## 当前判断与下一步','',
        '当前证据比前两轮更支持直接蒸馏共享页质量，并说明query均值之外的变化信息有用；但这只是一个对角方差构造的结果，不能推广为所有方差/协方差编码都有效。质量目标仍忽略V的方向和选中集合条件，且输出NMSE并非可见视频质量指标。',
        '', f'纯缓存K²（本计时同时缓存log-count）仅省约{b["variance_dynamic_k2"]-b["variance_cached_k2"]:.3f}ms/层调用，而全层FP32 K²多占253.125MiB。按本次研究实现的测量，不宜将这一缓存作为优先优化；可先考虑融合均值/方差归约与路由计算，再实际测端到端收益。',
        '', 'mass_mean无需增加在线方差算子，是较低成本候选；mass_var可作为误差更低但路由更贵的候选。下一阶段优先让这两者与硬均值在真实稀疏kernel和30秒闭环生成中比较，而不是继续只扩大离线loss消融。此前原仓库优化实现的正加速结果不受本次PyTorch eager局部计时否定。',
        '', f'mass_var有{sum(v==6 for v in epochs["mass_var"].values())}/30层被选为最后epoch，6epoch并不等价充分收敛；延长训练、更多训练数据或seed的收益尚未验证。所有本轮超参数已经固定，不再用这批确认结果反复调参后宣称独立测试。','']
    (ROOT/'report.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(total=total,comparisons=comparisons,alpha=summary['alpha']|{'values':'see summary.json'},control_delta=control_delta),indent=2))
if __name__=='__main__':run()
