import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_grad_20261031')
def main():
    smoke=json.loads((ROOT/'smoke.json').read_text());train=[json.loads(p.read_text()) for p in sorted((ROOT/'training').glob('*.json'))];events=[r for x in train for r in x['rows']]
    selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else None
    videos=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance'/'early').glob('*/*.json'))]
    cost={}
    if train:
        steady=[x for p,x in zip(sorted((ROOT/'training').glob('*.json')),train) if p.stem!='e0_train_city_walk_s0'] or train
        cost=dict(extra_fraction=sum(x['extra_training_seconds'] for x in steady)/sum(x['base_forward_seconds'] for x in steady),median_backward_ms=statistics.median(r['backward_seconds'] for r in events)*1000,max_extra_peak_gib=max(r['extra_peak_bytes'] for r in events)/2**30,clean_cases=sum(x['telemetry']['no_observed_foreign_processes'] for x in steady),cases=len(steady))
    lines=['# 最终x0损失真实反传路由参数','','## 这次确实更新了路由权重','','只有最终clean latent/x0 NMSE损失。torch.autograd沿输出head、最后block的FFN、cross-attention、残差、self-attention输出投影回到第29层的168个路由系数；Adam真实更新这些系数。视频模型参数全部冻结且grad=None。不同于之前只按loss选择有限门控配置。','','硬top-k不可导：前向仍走原生硬换页，反向用sigmoid(add/drop分数差)×两候选attention输出差构造STE。其forward严格为原生结果，backward是有偏代理，不对候选索引或冻结K/V求导。当前只反传最后block后半段，不跨多个Transformer block或去噪步；这是链路与成本pilot，不是完整两步BPTT。','','## 梯度证据','',f'- 路由梯度范数：{smoke["route_grad_norm"]:.6g}。',f'- 将最终输出误差方向反转，路由梯度余弦：{smoke["opposite_output_error_gradient_cosine"]:.6f}。',f'- Adam更新参数范数：{smoke["route_parameter_update_norm"]:.6g}。',f'- smoke前向与原生硬路由逐元素一致：{smoke["hard_forward_bit_exact"]}；所有视频模型参数grad=None。',f'- 已提交训练事件：{len(events)}/64，均检查梯度非零且有限、参数更新非零、history无额外commit。','']
    if cost:lines += [f'按已完成case汇总，除首次预热外，额外训练步骤耗时约{cost["extra_fraction"]:.2%}（teacher＋带STE的student前向＋backward，相对原有45次forward），反向中位数{cost["median_backward_ms"]:.3f}ms/事件，记录到的单事件额外峰值显存最大{cost["max_extra_peak_gib"]:.3f}GiB。{cost["clean_cases"]}/{cost["cases"]}个case未观测其他GPU进程。同步插桩/共享GPU限制解释，不能外推全层或两步反传。','']
    lines+=['## 训练与验证','','8个train场景×2轮，chunk2/5×step0/3共64次更新；LR0.01，初始化±0.3信赖范围。4个val场景使用固定初始化路由轨迹，同输入比较初始/学习后的最终x0误差。以验证选epoch，不使用回归视频选参。','']
    if selection:
        lines += [f'选择epoch{selection["epoch"]}（0-based）。','','| epoch | 初始x0 NMSE | 学习后x0 NMSE | 平均逐事件误差比 |','|---|---:|---:|---:|']
        for r in selection['rows']:lines.append(f'| {r["epoch"]} | {r["initial_nmse"]:.8f} | {r["trained_nmse"]:.8f} | {r["mean_ratio"]:.8f} |')
    lines+=['','## 前段闭环验收','','同模型、seed1001、KV上限及稳态41/410页预算；原始RoPE不变。RGB[21,101)，LPIPS越低越好。off=全部不修正；initial=最后层用训练前路由；trained=最后层用输出梯度训练后的路由。必须区分超过initial与超过off。','','| 场景 | 配置 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---:|---:|---:|']
    for r in videos:
        q=r['quality'];lines.append(f'| {r["case"]} | {r["mode"]} | {q["lpips"]:.5f} | {q["ssim"]:.5f} | {q["psnr"]:.3f} |')
    lines+=['','梯度存在和代理loss可优化不保证离散换页的真实loss下降。smoke的一次Adam后hard loss还略升，已经记录，不能把STE当作精确梯度。单层容量有限，teacher又使用全模型dense当前步，部分误差来自不可训练前缀；此实验不能判断全层输出监督上限。训练工作显存单列，推理常驻KV及辅助量5,095,914,800B未超限。详见[协议](PROTOCOL.zh.md)。']
    if len(videos)==6:
        lines+=['','## 本轮结论','','真实反传链路成立：64次更新都有非零梯度和非零参数变化，保存的checkpoint只有第29层swap系数变化（变化范数约1.986）；64次带梯度forward与原生hard x0的NMSE均为0。验证选中的epoch1平均逐事件输出误差比为0.998976，即约0.10%改善。','','视频：篮球相对initial从0.52249改善到0.51458，但仍略差于off的0.51334；蝴蝶相对initial从0.48864略退化到0.48945，优于off的0.49547。因此不能宣布稳定质量收益，也不能把蝴蝶相对off的优势全部归因于本次梯度训练，因为initial本来就更好。本轮没有进入完整30秒速度验收或路由kernel优化。','','这不是输出梯度方法上限：仅168个参数/最后层，前缀误差无法被充分修正；STE候选和margin代理有偏，单次更新不保证hard loss下降。后续若扩展到更早层，必须先补齐后续稀疏self-attention反向。原Runtime.forward和Triton attention没有autograd，只去掉外层no_grad会遗漏attention路径，即使残差仍有梯度也不能算完整梯度。','','## 对比视频','','左上旧64.88%稳定方案；右上90%off；左下最后层initial；右下最后层输出梯度trained。相同帧21–100，未调色。','','- [篮球](comparisons/basketball_court_early90.mp4)','- [蝴蝶](comparisons/butterfly_garden_early90.mp4)']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(smoke=smoke,training_cases=len(train),gradient_updates=len(events),cost=cost,selection=selection,videos=videos))
if __name__=='__main__':main()
