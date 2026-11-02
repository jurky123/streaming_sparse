import json
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_gain_20261031')
def main():
    selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else None
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance'/'early').glob('*/*.json'))]
    lines=['# 补偿感知路由实验','','## 路由瓶颈判断','','探针QK主乘加1.451 GFLOPs，dense QK+AV为742.895 GFLOPs，当前10%精确部分为74.290 GFLOPs。探针乘加仅占dense的0.1953%、精确部分的1.953%。这排除了“路由主乘加天然更大”这一解释，但不包含归约、topk、摘要维护和搬运。','','上一轮实测joint路由3.000ms/次，而精确attention+补偿0.675ms/次，均为共享GPU插桩。当前FP32路径、完整logits/特征落地、多次归约与启动使理论小计算量没有兑现。尚未做细粒度roofline分解，不能声称已确定各因素百分比或极限延迟。先验收有效路由目标，再优化实现。','','## 本轮变化','','冻结原始V和K分组，用真实“同时换入/换出”后的补偿误差监督14维上下文打分。原始基线/student轨迹交替训练，有界分数修正，最多一页交换，同KV上限与41/410精确页预算。数学检查通过，teacher不在部署阶段使用。','','## 验证选择','']
    if selection:
        lines += [f'验证选择修正强度：{selection["strength"]}。仅按局部NMSE选择，不代表视频验收。','','| 验证轨迹 | 0 | 0.25 | 0.5 | 1.0 |','|---|---:|---:|---:|---:|']
        for source,m in selection['by_source'].items():lines.append('| '+source+' | '+' | '.join(f'{m[s]:.6f}' for s in ['0.0','0.25','0.5','1.0'])+' |')
    else:lines.append('训练/验证进行中，尚无选参结果。')
    lines+=['','## 前段闭环回归','','未压缩RGB[21,101)，seed1001，LPIPS越低越好。全部预设强度均报告；篮球与蝴蝶未参与本轮训练/选参，属于既有回归场景。','','| 场景 | 强度 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---:|---:|---:|']
    for r in rows:
        q=r['quality'];lines.append(f'| {r["case"]} | {r["mode"]} | {q["lpips"]:.5f} | {q["ssim"]:.5f} | {q["psnr"]:.3f} |')
    lines+=['',f'前段完成{len(rows)}/8。质量收益需要同时与匹配90%基线、旧64.88%稳定方案比较；旧稳定LPIPS为篮球0.25812、蝴蝶0.14104。不得把局部误差改善当成闭环质量改善。','', '训练每case原子checkpoint，模型/优化器一起恢复；验收每视频原子提交。目录上限1GiB，不保存QKV。详见[协议](PROTOCOL.zh.md)。']
    if len(rows)==8:
        lines+=['','## 本轮结论','','验证集选出的强度1.0，在原始基线轨迹上局部NMSE下降约2.39%，在student轨迹上下降约2.22%。student验证的29层中27层改善，25/29层略退化。零修正的两个视频latent均与上一轮匹配baseline逐元素相同，排除了baseline实现漂移。数学公式与checkpoint恢复检查通过。','','前段结果仍未获得两个场景一致的保真度收益，本轮不通过质量门槛。局部收益能迁移到student轨迹，说明上一轮分布迁移问题有所缓解；但其幅度不足以改善最终视频，单次换页和低维线性打分也限制了纠错能力。teacher在训练样本筛选候选上有更大局部收益，只是离线诊断，不代表可部署收益上限。','','按“先算法、后实现”的顺序，本轮没有开始路由kernel融合，也没有对质量未过关的版本重复完整30秒测速。下一步应以短段去噪输出/latent偏差验证局部目标与闭环目标的联系，再决定扩展打分器或增加修正步数；不能仅凭训练局部NMSE继续加大模型。','','## 对比视频','','左上旧64.88%稳定方案，右上90%基线，左下90%修正强度0.5，右下90%强度1.0。相同RGB帧21–100，未调色。','','- [篮球](comparisons/basketball_court_early90.mp4)','- [蝴蝶](comparisons/butterfly_garden_early90.mp4)']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(selection=selection,early=rows))
if __name__=='__main__':main()
