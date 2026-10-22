"""Summarize fixed train-subset fits without reading Val/Test tensors."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--round',required=True)
    args = p.parse_args()
    root = Path(args.round).resolve()
    protocol = json.loads((root/'protocol.json').read_text())
    steps = protocol['steps']
    results, contrasts = {}, {}
    for fusion in ('concat','film'):
        results[fusion] = {}
        for arm in 'ABCD':
            d = root/'runs'/fusion/arm
            first = json.loads((d/'evaluation_000000.json').read_text())
            last = json.loads((d/f'evaluation_{steps:06d}.json').read_text())
            results[fusion][arm] = dict(
                initial_learned=first['learned']['macro_pair_relative_error'],
                final_learned=last['learned']['macro_pair_relative_error'],
                final_soft=last['soft']['macro_pair_relative_error'],
                same_partition_mass=last['mass']['macro_pair_relative_error'],
                initial_partition_mass=last['initial_mass']['macro_pair_relative_error'],
                learned_over_mass=last['learned_over_same_partition_mass'],
                learned_over_initial_mass=last['learned_over_initial_mass'],
                learned_improvement_percent=100*(1-last['learned']['macro_pair_relative_error']/first['learned']['macro_pair_relative_error']),
                fitted_to_mass_10pct=last['fitted_to_mass_10pct'],
                pair_count_within_25pct_mass=last['pair_count_within_25pct_mass'],
                mean_repair_fraction=last['mean_repair_fraction'],
                mean_partition_change_from_initial=last['mean_partition_change_from_initial'],
                per_pair_learned=last['learned']['per_pair'],per_pair_mass=last['mass']['per_pair'],
                budget_violations=last['budget_violations'])
        r = results[fusion]
        contrasts[fusion] = {
            'B_over_A_learned_error':r['B']['final_learned']/r['A']['final_learned'],
            'C_over_B_learned_error':r['C']['final_learned']/r['B']['final_learned'],
            'D_over_C_learned_error':r['D']['final_learned']/r['C']['final_learned'],
        }
    reportdir = root/'reports'
    reportdir.mkdir(exist_ok=True)
    (reportdir/'results.json').write_text(json.dumps(dict(protocol=protocol,results=results,contrasts=contrasts),indent=2)+'\n')
    lines = ['v0.3 修复实验：固定 12-snapshot 多任务学习定位', '',
        '设计：同一 train prompt、generation seed0、history9360，3 layers × 4 physical heads，',
        '每个 snapshot 四个 diffusion steps，固定 tile0、keep0.2；每次更新4 snapshots，',
        f'各组 {steps} 次 AdamW 更新、相同初始化和采样顺序，单 GPU 顺序运行。',
        'A=冻结初始F/partition；B=固定partition、F可训练；C=原joint训练；D=C+独立pair score。',
        '指标：先在每个 pair 内汇总4 steps的 SSE/energy，再对12 pairs平均。',
        '训练目标：每个 step/tile normalized SSE 的等权均值；与上述汇总口径略有不同。',
        'Mass 是同一分区的参考策略，不是最优下界。10%阈值仅为预注册的拟合诊断，不是泛化门槛。', '']
    for fusion in ('concat','film'):
        lines += [fusion.upper(), '组 | Learned | Soft | 同分区Mass | 初始分区Mass | Learned/Mass | 12 pairs中接近Mass数']
        for arm,r in results[fusion].items():
            lines.append(f"{arm} | {r['final_learned']:.8f} | {r['final_soft']:.8f} | {r['same_partition_mass']:.8f} | {r['initial_partition_mass']:.8f} | {r['learned_over_mass']:.3f} | {r['pair_count_within_25pct_mass']}/12")
        r = results[fusion]
        c = contrasts[fusion]
        lines += [f"B/A={c['B_over_A_learned_error']:.3f}；C/B={c['C_over_B_learned_error']:.3f}；D/C={c['D_over_C_learned_error']:.3f}（均为Learned绝对误差比）。",
                  f"C repair={r['C']['mean_repair_fraction']:.3f}，经最优标签匹配后的分区变化={r['C']['mean_partition_change_from_initial']:.3f}。",
                  f"D repair={r['D']['mean_repair_fraction']:.3f}，分区变化={r['D']['mean_partition_change_from_initial']:.3f}。"]
        if r['A']['fitted_to_mass_10pct']:
            lines.append('A 在这一小训练集合达到预注册 Mass 拟合阈值：共享 Decoder 可在固定F/partition下学习这些任务。')
        else:
            lines.append('A 未达到 Mass 拟合阈值：单靠冻结分区尚不能消除多任务拟合问题，不能将根因单独锁定为 clustering。')
        if r['B']['fitted_to_mass_10pct'] and not r['C']['fitted_to_mass_10pct']:
            lines.append('B 达标而 C 未达标：该受控比较支持 joint partition 优化造成困难；尚不能进一步区分 STE、repair 与边界变化各自的贡献。')
        if r['A']['fitted_to_mass_10pct'] and not r['B']['fitted_to_mass_10pct']:
            lines.append('A 达标而 B 未达标：该集合中，固定分区下训练F仍使拟合更困难；需要重复种子与学习率对照。')
        if r['C']['fitted_to_mass_10pct']:
            lines.append('C 也能在该固定集合达到阈值：当前分区训练路径并非必然无法多任务拟合；这不能证明原大集合的优化问题已解决。')
        lines += ['']
    lines += ['证据与限制',
        '1. 所有结论限于一个 train prompt / generation seed / history 与 router seed0；未做 Val/Test 或泛化结论。',
        '2. 四组相同初始有效 score、hard partition、hard mask通过逐组真实QKV数值检查；A/B分区训练期间逐次检查不变。',
        '3. D 只与 C 配对；128→1 rank-1 已有完整线性表达范围，D 改变共享/因子化/优化动力学，不能直接当作容量证据。',
        '4. C/D 同分区Mass可随分区变化；同时报告固定初始Mass，避免参考策略退化带来虚假“达标”。',
        '5. 非线性训练轨迹允许变量交互；一次种子对照不能证明某机制是唯一根因。',
        '6. 不扩大heads，不进入diffusion loss，不使用Mass/Oracle监督，不新增正则/位置特征。',
        '7. final.pt固定取最后更新，不使用Val挑选；curve.jsonl与每100更新原始48-case JSON完整保存。']
    (reportdir/'analysis_report.txt').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors = dict(A='#4477aa',B='#228833',C='#cc6677',D='#aa3377')
    for col,fusion in enumerate(('concat','film')):
        for arm in 'ABCD':
            d = root/'runs'/fusion/arm
            initial = json.loads((d/'evaluation_000000.json').read_text())
            curves = [json.loads(line) for line in (d/'curve.jsonl').read_text().splitlines()]
            xs = [0]+[r['updates'] for r in curves]
            ys = [initial['learned']['macro_pair_relative_error']]+[r['learned']['macro_pair_relative_error'] for r in curves]
            axes[0,col].plot(xs,ys,label=arm,color=colors[arm])
            axes[1,col].plot(xs,[initial['learned_over_same_partition_mass']]+[r['learned_over_same_partition_mass'] for r in curves],label=arm,color=colors[arm])
        axes[0,col].axhline(results[fusion]['A']['initial_partition_mass'],color='black',ls='--',lw=1,label='Initial Mass')
        axes[1,col].axhline(1,color='black',ls='--',lw=1)
        axes[1,col].axhline(1.10,color='gray',ls=':',lw=1)
        axes[0,col].set_title(fusion.upper()+' — fixed 12 train snapshots')
        axes[0,col].set_ylabel('Macro pair attention error')
        axes[1,col].set_ylabel('Learned / same-partition Mass')
        for ax in axes[:,col]:
            ax.set_xlabel('Optimizer updates')
            ax.set_yscale('log')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(reportdir/'causal_repair.png',dpi=180)
    fig.savefig(reportdir/'causal_repair.pdf')
    print(json.dumps(dict(results=results,contrasts=contrasts),indent=2))


if __name__ == '__main__':
    main()
