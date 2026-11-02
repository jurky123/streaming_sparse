"""Independent confirmation, no test-based method or layer selection."""
import csv,json
import numpy as np
import torch
from .tile_common import ROOT,cases,CHUNKS,guard
from .tile_router import METHODS
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
                cell[r['method']]={f:r[f][h] for f in ['nmse','sse','energy','max_query_nmse']}
    keys=sorted(data)
    expected={(c['name'],c['seed'],ch,s,l,h) for c in cases('test') for ch in CHUNKS for s in range(4) for l in range(30) for h in range(12)}
    assert set(keys)==expected and len(keys)==51840
    assert all(set(v)==set(methods) for v in data.values())
    a={m:{f:np.array([data[k][m][f] for k in keys]) for f in ['nmse','sse','energy','max_query_nmse']} for m in methods}
    total={m:dict(nmse=float(v['nmse'].mean()),energy_nmse=float(v['sse'].sum()/v['energy'].sum()),p95=float(np.quantile(v['nmse'],.95)),worst_query=float(v['max_query_nmse'].max())) for m,v in a.items()}
    groups={}
    for name,cols in [('prompt',[0]),('seed',[1]),('chunk',[2]),('step',[3]),('layer',[4]),('head',[5]),('layer_head',[4,5])]:
        groups[name]=[]
        for g in sorted({tuple(k[i] for i in cols) for k in keys}):
            mask=np.array([tuple(k[i] for i in cols)==g for k in keys]);row=dict(group='/'.join(map(str,g)),n=int(mask.sum()))
            row.update({m:float(a[m]['nmse'][mask].mean()) for m in methods});groups[name].append(row)
        with (ROOT/f'test_by_{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(groups[name][0]));w.writeheader();w.writerows(groups[name])
    pairs=[(m,'mean') for m in METHODS]+[('random_tile','random_query'),('anchor_query','random_query'),('anchor_tile','anchor_query'),('anchor_tile','random_tile')]
    rng=np.random.default_rng(42);draw=rng.integers(0,6,(10000,6));comparisons={}
    for m,b in pairs:
        pm=np.array([r[m] for r in groups['prompt']]);pb=np.array([r[b] for r in groups['prompt']])
        samples=100*(pm[draw].mean(1)/pb[draw].mean(1)-1)
        comparisons[f'{m}_vs_{b}']=dict(relative_pct=100*(total[m]['nmse']/total[b]['nmse']-1),ci95_pct=np.quantile(samples,[.025,.975]).tolist(),
            prompt_regressions=int((pm>pb).sum()),layer_regressions=sum(r[m]>r[b] for r in groups['layer']),layer_head_regressions=sum(r[m]>r[b] for r in groups['layer_head']),
            head_event_regressions=float((a[m]['nmse']>a[b]['nmse']+1e-12).mean()),energy_change_pct=100*(total[m]['energy_nmse']/total[b]['energy_nmse']-1))
    epochs={m:{} for m in METHODS}
    for folder in [ROOT/'train_00_15',ROOT/'train_15_30']:
        st=json.loads((folder/'status.json').read_text());assert st['status']=='completed' and all(v==576 for v in st['updates'].values())
        saved=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
        for m in METHODS:
            for l in saved['models'][m]:epochs[m][l]=min((c['scores'][m][l],c['epoch']) for c in saved['curves'])[1]
    for name in ['router_checks.json','training_resume_checks.json','training_ledger_checks.json']:
        assert json.loads((ROOT/name).read_text())['passed']
    summary=dict(total=total,comparisons=comparisons,groups=groups,selected_epochs=epochs,
        coverage=dict(head_events=51840,query_head_measurements=1658880,replay_layer_steps=audits),disk=guard())
    atomic(ROOT/'summary.json',summary)
    lines=['# Tile一致性与硬均值起点：独立确认实验','',
        '同冻结模型、原始RoPE，无rebase。固定5chunk历史KV、72个选中历史page、每页4代表，不跨page、不随距离减代表。复用8训练/4验证prompt；新6个测试prompt×2seed(404/505)，上一轮测试集未进入本轮训练/选择/评估。30层×12heads，chunk5/20/40，四步去噪；每方法51,840个head-events、1,658,880个query-head测量。',
        '', '四组均6epoch，每层576次更新，验证集独立选择每层best且允许epoch0。所有组训练query一致：8个tile×每tile4个实际query。验证/测试使用32个均匀query及其64-query tile均值路由。',
        '', 'random_query：原随机mixture+实际query的split目标；random_tile：同mixture，但路由质量与排序使用tile均值预测、以tile内四个真实query监督；anchor_query/anchor_tile：分别使用相同目标，但改为response硬子组内的有界可学习加权聚合，零初始化等于硬均值。V条件拟合仍使用实际query。',
        '', 'anchor同时改变初始化和聚合器约束，不能将差异纯归因初始化。固定count16是近似质量；加权聚合不是token质量守恒的soft分区。tile目标是路由一致性代理，未直接优化离散top-k的真实输出误差。',
        '', '| 方法 | 平均 query NMSE ↓ | 能量加权误差 ↓ | head-event P95 ↓ |','|---|---:|---:|---:|']
    for m in methods:
        r=total[m];lines.append(f'| {m} | {r["nmse"]:.8f} | {r["energy_nmse"]:.8f} | {r["p95"]:.8f} |')
    lines+=['', '验证集选择的epoch0层数：'+ '，'.join(f'{m}: {sum(v==0 for v in epochs[m].values())}/30' for m in METHODS)+'。anchor的epoch0等于硬均值，因此改善不能解释为全部30层都从学习中获益。']
    lines+=['','## 成对消融','', '| 对比（前者相对后者） | NMSE变化 | prompt聚类95%区间 | 退化场景 | 退化层 | 能量加权变化 |','|---|---:|---:|---:|---:|---:|']
    for key,c in comparisons.items():
        lo,hi=c['ci95_pct'];lines.append(f'| {key} | {c["relative_pct"]:+.3f}% | [{lo:+.3f}%, {hi:+.3f}%] | {c["prompt_regressions"]}/6 | {c["layer_regressions"]}/30 | {c["energy_change_pct"]:+.3f}% |')
    lines+=['','负值表示改善。bootstrap按六个prompt聚类，两seed先在场景内合并，不把head/query当独立样本；10,000次重采样、seed42。上述区间未做多重比较校正，作为探索性不确定性描述，不代表已检验训练seed鲁棒性。',
        '', '## 场景拆分','', '| 场景 | mean | random_query | random_tile | anchor_query | anchor_tile |','|---|---:|---:|---:|---:|---:|']
    for r in groups['prompt']:lines.append('| '+r['group']+' | '+' | '.join(f'{r[m]:.6f}' for m in methods)+' |')
    lines+=['','完整layer/head/chunk/step/seed拆分在test_by_*.csv，逐事件结果在test_worker*/。不根据本轮test选择回退层或调整超参数后继续称它为独立测试。',
        '', '## 验证与边界','',
        '- 全验证集初始状态检查：两种anchor与硬均值的34,560次head-event对比，NMSE差值均为0。见initial_baseline_checks.json。',
        '- 真实重放检查通过：anchor初始输出与硬均值一致、修改一page不改变其他page代表、tile内路由共享、四组梯度有限且非零。',
        '- 实际训练在4次更新后停止、恢复到8次；layer0四组权重与Adam状态与连续训练逐元素一致，见training_resume_checks.json。',
        '- 测试所有4,320个layer-step事件中，保存的4个query与重放逐元素一致。评估实际选中原KV的输出，沿用上一轮通过直接KV检查的评估器。',
        f'- 新目录实际占用约{summary["disk"]["run_mib"]:.1f} MiB，4GiB上限。旧训练/验证数据与字典通过只读使用的symlink复用，未复制；原数据另占约1GiB。每chunk原子提交、滚动checkpoint、哈希校验、文件锁、恢复需重放clean前缀。',
        '- 只完成离线局部替换；不证明闭环30秒视频效果或端到端速度收益。原始/选中KV预算一致，但额外摘要与临时空间不包含在同物理总KV预算内。',
        '- 四代表FP32 K/V全层payload仍约506.25MiB（BF16部署设想253.125MiB）；anchor额外参数约0.703MiB，固定counts可省存但本实验未实现该部署优化。无新增在线query网络，主要学习计算在缓存预处理。','']
    diag=json.loads((ROOT/'val_diagnostics'/'summary.json').read_text())
    drop=100*(1-diag['nmse']['shared_actual_mass']/diag['nmse']['exact_tile_query'])
    lines+=['## 验证集的路由目标诊断','',
        '另外在4个验证prompt、6个预定层(0/4/14/17/24/29)、3个chunk和4步去噪上，抽查288个layer-step事件，每head取8个tile×4query。该诊断在看到中途验证走势后添加，未用新测试集，不参与训练或checkpoint选择。',
        '', '| 选页分数来源 | 实际稀疏输出NMSE |','|---|---:|']
    for m,value in diag['nmse'].items():lines.append(f'| {m} | {value:.8f} |')
    lines+=['',f'将精确tile均值query的历史页质量，改为tile内四个真实query的平均完整注意力质量，误差降低{drop:.2f}%，约替换{diag["replaced_fraction"]:.1%}的选中页。平均route/query范数比为{diag["route_norm_ratio"]:.3f}。',
        '', 'mean_proto使用四个硬均值代表；exact_tile_query使用精确K计算tile均值query注意力；shared_actual_mass对四个真实query的归一化完整注意力质量取平均并共享选页；per_query_mass允许每个query单独按精确质量选页。全部选72历史页且当前KV全保留。',
        '', '这些精确分数需要访问完整KV，不能直接部署为稀疏加速；按质量选页不是输出误差最优oracle，也不是严格误差下界。它们主要诊断query聚合/选页代理的缺口，不能将验证集潜力值当确认集可实现收益。',
        '', '当前tile训练仍用实际query的逐页logZ和排序监督均值query预测，并未直接学习shared_actual_mass。后续应优先比较直接蒸馏这种共享质量分数，以及增加tile内query变化的摘要（例如方差），再考虑选择集合相关的V误差。仅靠均值query可能丢失决定页重要性的信息；这属于待验证假设。','']
    intro=['',
        '本轮结论：硬均值起点的受限学习，相对随机聚合有一致的平均NMSE改善，但相对硬均值基线的收益仍小且不确定；当前tile目标没有稳定独立收益。',
        '', 'anchor_query相对random_query降低1.04%，anchor_tile相对random_tile降低0.88%，两个对照都是6/6个新场景改善。相对硬均值，anchor_query/anchor_tile仅降低0.42%/0.44%，场景聚类区间仍跨零。anchor_tile的能量加权误差相对硬均值降低2.45%，但它有18/30层被验证集选为初始硬均值，不应宣称所有层都需要学习。',
        '', '仅替换tile监督目标：random_tile相对random_query改善0.18%，anchor_tile相对anchor_query改善0.02%，两项区间均跨零。随机聚合的两组均未超过硬均值基线。当前证据不足以据此全层部署或承诺视频质量收益。','']
    lines[1:1]=intro
    lines+=['## 聚合与后续实验的具体含义','',
        '每个64-token page已有四个16-token response硬子组。对每组r，先计算中心化K，再用 s_i=2*tanh(w_r·(K_i−mean(K_r))/2)，对该组16个token做softmax得到a_i，K代表=Σa_i K_i，V代表=Σa_i V_i；count固定16。w初始化为0，因而初始a_i=1/16，严格回到硬均值。分配器输入只有K，V通过加权代表及训练loss影响梯度，尚未直接作为分配特征。',
        '', '这个方案同时保留已有分组、限制学习自由度，并提供精确基线起点；本轮未把这三个因素进一步分离。下一步若研究初始化本身，需要同一参数化下的初始化消融，不能直接将本轮收益全算给初始化。',
        '', '优先级建议：先把shared_actual_mass作为离线teacher标签，训练真正对应共享tile决策的页分数；再比较只用query均值与增加query方差/少量query代表的输入。均值输入可能缺少区分tile内分布的信息，仅更换loss未必够。随后再加入当前已选集合相关的V输出误差，评估小幅换块的空间。',
        '', '这些都是未完成的后续实验方向；本轮的30%仅是验证集精确打分诊断潜力。下一轮改法需另留新的确认集，或明确将本轮新测试集降为开发集。','']
    summary['validation_diagnostic']=diag
    atomic(ROOT/'summary.json',summary)
    (ROOT/'report.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(total=total,comparisons=comparisons),indent=2))
if __name__=='__main__':run()
