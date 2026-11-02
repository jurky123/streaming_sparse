import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_output_20261031')
def main():
    samples=[json.loads(p.read_text()) for split in ['train','val'] for p in (ROOT/split).glob('*.json')]
    stable=[r for r in samples if r['case']['id']!='train_city_walk_s0']
    summary=dict(samples=len(samples),extra_calls_per_case=7,base_calls_per_case=45)
    if stable:
        summary.update(pooled_extra_fraction=sum(r['extra_forward_seconds'] for r in stable)/sum(r['base_forward_seconds'] for r in stable),min_extra_fraction=min(r['incremental_forward_cost'] for r in stable),max_extra_fraction=max(r['incremental_forward_cost'] for r in stable),median_extra_fraction=statistics.median(r['incremental_forward_cost'] for r in stable),clean_cases=sum(r['telemetry']['no_observed_foreign_processes'] for r in stable))
    selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else None
    videos=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance'/'early').glob('*/*.json'))]
    lines=['# 最终去噪输出监督：成本试验','','## 训练会不会很慢','','直接对完整视频反传会明显增加成本。冻结参数不代表无需后续层反传；若要求输出损失对路由权重的梯度，还需要保留/重算相关激活。当前先做无需反传的硬策略校准，避免提前支付该成本。','','每场景9chunks原有45次前向，只抽一个去噪步，增加5个候选、1个dense teacher、1个source重复审计，共7次，即前向次数+15.56%。只存JSON，不存QKV或反传激活。source重复逐元素一致，缓存索引、history写入位置及clean commit均未改变。','']
    if stable:lines += [f'已采集{len(samples)}/12个场景；排除首次预热后，额外前向耗时合计约{summary["pooled_extra_fraction"]:.2%}，各场景范围{summary["min_extra_fraction"]:.2%}–{summary["max_extra_fraction"]:.2%}，中位数{summary["median_extra_fraction"]:.2%}。{summary["clean_cases"]}/{len(stable)}场景未观测到其他GPU进程。同步插桩/共享GPU影响仍存在；此比例仅描述本校准采集，不是端到端训练通用倍数。','']
    lines+=['## 监督定义','','teacher用同一个noisy输入、同一个student历史KV，当前所有self-attention dense；比较模型最终clean latent/x0，而非单层attention输出。它依然是条件于student历史的短程teacher，不是完整dense轨迹。六种策略off/half/all/early/middle/late，对应已有路由的全关、半幅、全开、前三/中/后三分之一层开启。路由权重及V冻结，按训练集输出NMSE选择硬门控。','']
    if selection:
        lines += [f'训练集选择：**{selection["policy"]}**。验证集仅用于只读评估，不参与选择。','','| 数据 | off | half | all | early | middle | late |','|---|---:|---:|---:|---:|---:|---:|']
        for split,m in selection['means'].items():lines.append('| '+split+' | '+' | '.join(f'{m[k]:.6f}' for k in ['off','half','all','early','middle','late'])+' |')
    lines+=['','## 前段闭环验收','','未压缩RGB[21,101)，seed1001；选参不使用篮球/蝴蝶。','','| 场景 | 策略 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---:|---:|---:|']
    for r in videos:
        q=r['quality'];lines.append(f'| {r["case"]} | {r["mode"]} | {q["lpips"]:.5f} | {q["ssim"]:.5f} | {q["psnr"]:.3f} |')
    lines+=['','此试验只回答输出监督采样成本及粗粒度策略选择，不能替代可学习路由权重的输出梯度训练。样本覆盖每场景仅一个step，不能据此声称输出监督已解决闭环质量。固定90%稀疏调用预算与KV上限，未进行路由kernel优化。详见[协议](PROTOCOL.zh.md)。']
    if selection and len(videos)==4:
        lines+=['','## 试验结论','','本次选出的all策略与上一轮全层强度1.0相同。两个场景的off/all共4条新生成latent均与已有结果逐元素一致，因此直接复用原有VAE/指标结果，没有重复编码视频。all的前段LPIPS仍为篮球0.52386、蝴蝶0.50804，均差于off的0.51334、0.49547。','','单步输出损失在验证集改善约2.69%，但仍不能保证闭环质量；训练集平均改善仅0.073%，样本不足以说明稳健收益。监督已经从单层attention扩展到最终x0，但完整历史仍是student历史，且没有多步展开。下一步应先增加少量连续去噪步、固定同噪声的短程teacher，检查误差累积，而不是直接训练整段30秒。','','成本结论：稀疏采样的无反传输出校准可控制在约15.5%的额外forward时间；这不是输出梯度训练成本。真正训练路由权重时还需硬top-k的监督/梯度设计，以及额外反传或更细粒度候选评估，必须另行测量。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(cost=summary,selection=selection,videos=videos))
if __name__=='__main__':main()
