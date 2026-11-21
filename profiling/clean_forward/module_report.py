"""Summarize paired module interventions. Negative deltas indicate lower error."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent/'module_sensitivity'
cases=['summary','summary_scope','summary_clean','risk','risk_scope','risk_clean','reader','reader_scope','reader_local_best','reader_local_worst']
lines=['# intersection_fast 真实模块权重消融','',
       '5 秒测试使用 city_walk 和 short_721，各用 seed 721、722。局部实验在每个稀疏层计分后回填稠密注意力输出；视频实验使用正常稀疏闭环。LPIPS 使用解码后的 80 帧，latent MSE 从第四个 chunk 起计算。百分数均以原权重为基准，负值表示误差降低。','']
def load_scores(path):
    rows=json.loads(path.read_text())
    assert len(rows)==4
    return {name:np.array([r[name] for r in rows]) for name in ('mse','lpips')}
def delta_local(x,b):
    return float(100*(x[...,0].sum()/b[...,0].sum()-1))
for mode in ('dense','sparse'):
    b=load_scores(root/f'{mode}_off.json')
    base=np.load(root/f'{mode}_off_local.npy')
    lines += [f'## {mode} clean','','| 干预 | 全部稀疏调用局部 SSE Δ% | 非零变动单元局部 SSE Δ% | latent MSE 配对 Δ% | LPIPS 均值 | LPIPS 改善条数 |',
              '|---|---:|---:|---:|---:|---:|',f"| 原权重 | 0 | — | 0 | {b['lpips'].mean():.5f} | — |"]
    axes={}
    for case in cases:
        file=root/f'{mode}_{case}.json'
        if not file.exists():continue
        x=load_scores(file);a=np.load(root/f'{mode}_{case}_local.npy')
        assert np.array_equal(a[...,1],base[...,1]),case
        changed=a[...,0]!=base[...,0]
        affected=delta_local(a[changed],base[changed]) if changed.any() else 0.
        lines.append(f"| {case} | {delta_local(a,base):+.4f} | {affected:+.4f} | {(np.mean(x['mse']/b['mse'])-1)*100:+.2f} | {x['lpips'].mean():.5f} | {sum(x['lpips']<b['lpips'])}/4 |")
        axes[case]={}
        for axis,label in enumerate(('step','layer','head')):
            table={}
            for i in range(a.shape[axis]):
                v=np.take(a,i,axis);ref=np.take(base,i,axis)
                if ref[...,0].sum():table[i]=delta_local(v,ref)
            axes[case][label]=table
    (root/f'{mode}_local_axes.json').write_text(json.dumps(axes,indent=2))
    lines+=['']
lines+=['summary 使用历史条件 10% 微调池化权重，risk 使用逐步、层、头标定的四项正系数，reader 使用历史条件 10% 微调共享网络。summary_scope 为七层三头，risk_scope 另限定 step 0，reader_scope 只在第 5 层第 3 头读取替换的 reader 编码。summary_clean 对 clean 阶段及其提交、锚点摘要使用替换权重。risk_clean 只在 clean 步使用替换系数。非零变动单元由已汇总的 step/layer/head 局部 SSE 是否发生改变界定。']
lines+=['','## 30 秒视频复验','',
        '按 5 秒 LPIPS 平均改善且至少 3/4 条改善筛选，每种 clean 模式取符合条件者中均值最低的一项。另对局部误差变化排名两端的 reader 单元追加诊断。复验使用同四条输入延长到 480 帧。sparse clean 原权重取同输入的上一轮未微调对照。',
        '', '| clean 模式及干预 | LPIPS 均值，原 → 新 | LPIPS 改善 | 逐轨迹相对 latent MSE 变化均值 |','|---|---:|---:|---:|']
for mode,case in [('dense','summary'),('sparse','summary_clean'),('sparse','reader_local_best'),('sparse','reader_local_worst')]:
    candidate=root/f'{mode}_{case}_30s.json'
    if not candidate.exists() or len(json.loads(candidate.read_text()))!=4:continue
    if mode=='dense':base=load_scores(root/'dense_off_30s.json')
    else:
        prior=json.loads((root.parent/'sensitivity_sparse_confirm_30s.json').read_text())['off']
        base={'mse':np.array(prior['mse']),'lpips':np.array(prior['lpips480'])}
    scores=load_scores(candidate)
    lines.append(f"| {mode} / {case} | {base['lpips'].mean():.5f} → {scores['lpips'].mean():.5f} | {sum(scores['lpips']<base['lpips'])}/4 | {100*(np.mean(scores['mse']/base['mse'])-1):+.2f}% |")
lines+=['','reader_local_best / reader_local_worst 分别只替换 step 0 的第 17 层第 3 头 / 第 27 层第 9 头。两者按本批固定输入上的 21 个 reader 活跃单元局部误差排名选取，用于诊断同批闭环排名。']
(root/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
