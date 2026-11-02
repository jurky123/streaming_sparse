import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
from .high_sparse_output_runtime import POLICIES
ROOT=Path('artifacts/high_sparse_twostep_20261031')
def main():
    data={s:[json.loads(p.read_text()) for p in sorted((ROOT/s).glob('*.json'))] for s in ['train','val']};samples=data['train']+data['val'];steady=[r for r in samples if r['case']['id']!='train_city_walk_s0']
    cost={} if not steady else dict(extra_fraction=sum(r['extra_forward_seconds'] for r in steady)/sum(r['base_forward_seconds'] for r in steady),median=statistics.median(r['incremental_forward_cost'] for r in steady),minimum=min(r['incremental_forward_cost'] for r in steady),maximum=max(r['incremental_forward_cost'] for r in steady),no_foreign_process_cases=sum(r['telemetry']['no_observed_foreign_processes'] for r in steady),cases=len(steady))
    selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else None
    schedule=json.loads((ROOT/'schedule_selection.json').read_text()) if (ROOT/'schedule_selection.json').exists() else None
    means={s:{h:{p:sum(r['rows'][0]['metrics'][p][h]['nmse'] for r in rs)/len(rs) for p in POLICIES} for h in ['one','two']} for s,rs in data.items() if rs}
    videos=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance'/'early').glob('*/*.json'))]
    lines=['# 两步去噪输出监督实验','','## 本轮做了什么','','从同一noisy输入与历史KV出发，teacher和candidate各自走两步。第二步使用本分支第一步预测加相同噪声；监督第二步全模型输出的clean latent，保留同一事件的单步误差作比较。没有将teacher第一步输出喂给student。','','视频模型、K/V摘要、路由系数冻结。本轮按输出损失拟合全局和分阶段硬门控，不是输出梯度直接更新路由权重。预设候选off/half/all/early/middle/late。训练8场景、验证4场景，每场景抽一个两步事件；验证不选参。','']
    if cost:lines += [f'额外14次前向/原有45次。排除首场景预热后，采样额外前向耗时合计{cost["extra_fraction"]:.2%}，中位数{cost["median"]:.2%}，范围{cost["minimum"]:.2%}–{cost["maximum"]:.2%}。{cost["no_foreign_process_cases"]}/{cost["cases"]}个场景未观测到其他GPU进程。同步插桩、共享GPU限制这些数字；不代表输出反传训练成本。','']
    lines+=['## 单步与两步输出误差','','两种loss的teacher/归一化不同，绝对值不宜跨列作改善比例；看同一loss内策略相对off的变化。','','| 数据 | horizon | off | half | all | early | middle | late |','|---|---|---:|---:|---:|---:|---:|---:|']
    for s,hs in means.items():
        for h,m in hs.items():lines.append('| '+s+' | '+h+' | '+' | '.join(f'{m[p]:.6f}' for p in POLICIES)+' |')
    if selection:lines += ['',f'按训练两步NMSE选择全局策略：**{selection["policy"]}**。']
    if schedule:
        lines += [f'按训练两步NMSE分阶段拟合：step0–1用 **{schedule["policies"][0]}**，step2–3用 **{schedule["policies"][1]}**。每组只有4个训练事件。','','| 数据 | off两步NMSE | schedule两步NMSE | 相对下降 |','|---|---:|---:|---:|']
        for s,m in schedule['evaluation'].items():lines.append(f'| {s} | {m["off"]:.6f} | {m["schedule"]:.6f} | {1-m["schedule"]/m["off"]:.2%} |')
    lines+=['','## 前段视频验收','','同模型、seed1001、KV上限和90%稀疏调用预算。质量用未压缩RGB[21,101)，LPIPS越小越接近dense。','','| 场景 | 策略 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---:|---:|---:|']
    for r in videos:
        q=r['quality'];lines.append(f'| {r["case"]} | {r["mode"]} | {q["lpips"]:.5f} | {q["ssim"]:.5f} | {q["psnr"]:.3f} |')
    lines+=['','teacher仅在当前两步dense，初始历史仍是student历史；没有跨chunk更新历史的teacher展开。短程损失仍不能等同视频质量。缓存分支隔离要求首步重放与正常生成的后一步均逐元素一致；历史commit和索引不变。常驻预算5,095,914,824B，不保存QKV或反传激活。每case/视频原子恢复，目录上限1GiB。详见[协议](PROTOCOL.zh.md)。']
    if len(videos)==6:
        lines+=['','## 结论','','本轮未通过质量验收。全局early和分阶段early/all都没有在两个回归场景上建立稳定保真度收益；没有进入路由kernel优化或完整30秒测速。训练两步loss略降但验证略升，视频也未改善，不能因为监督位置更下游就认定有效。','','同一批8个训练事件、5个非off候选共40组比较，有11组相对off的收益方向在one→two后反转。单步训练均值最佳为off，两步最佳为early。这说明后续去噪会改变候选排序，不能只看当前attention或当前x0；它并不证明两步目标足以预测视频质量。','','限制主要在于：本轮只是六个粗粒度候选的硬门控拟合，没有直接更新路由权重；每场景一个事件，训练/验证覆盖少；teacher只展开同一chunk内两步，没有教师历史和跨chunk误差监督。不能据此否定下游监督本身。下一步应把下游收益用于更细粒度路由参数学习，并扩展训练事件覆盖，再验证小段跨chunk历史误差。','','## 对比视频','','左上旧64.88%稳定方案；右上90%无修正基线；左下两步loss选择的global；右下两步loss拟合的schedule。相同帧21–100，未调色。','','- [篮球](comparisons/basketball_court_early90.mp4)','- [蝴蝶](comparisons/butterfly_garden_early90.mp4)']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(cost=cost,means=means,selection=selection,schedule=schedule,videos=videos))
if __name__=='__main__':main()
