import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/full_route_grad_20261031')
def main():
    smoke=json.loads((ROOT/'smoke.json').read_text());cap=json.loads((ROOT/'smoke_chunk5.json').read_text()) if (ROOT/'smoke_chunk5.json').exists() else None
    train=[json.loads(p.read_text()) for p in sorted((ROOT/'training').glob('*.json'))];events=[r for x in train for r in x['rows']]
    selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else None
    videos=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance'/'early').glob('*/*.json'))]
    status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {'status':'not_started'}
    cost={} if not events else dict(median_backward_seconds=statistics.median(r['backward_seconds'] for r in events),max_extra_peak_gib=max(r['extra_peak_bytes'] for r in events)/2**30,extra_fraction=sum(x['extra_training_seconds'] for x in train)/sum(x['base_forward_seconds'] for x in train))
    lines=['# 全29稀疏层路由：完整单步输出梯度训练','',f'当前状态：{status["status"]}；已提交{len(train)}/16个训练case、{len(events)}/64次梯度更新。','','## 先完成路由优化','','路由优化和两个场景的全长一致性/速度测试已完成，详见[优化验收](../route_graph_20261031/report.zh.md)。CUDA Graph路径用于本轮推理及训练前向。','','## 本轮的梯度路径','','训练1–29层的12×14路由系数，共4,872参数；第0层保持dense且无稀疏路由。视频模型参数冻结。最终x0损失通过所有后续self-attention的Q/K/V、当前K代表/当前V code、投影、FFN和cross-attention回传。历史KV及历史摘要停止梯度。仍是单个去噪步的全层反传，不是跨去噪步/跨chunk BPTT。','','前向使用原生稀疏+补偿kernel；反向在每个query tile组装选中真实token和未选页的4个K代表，代表带log16质量，重算FP32 memory-efficient SDPA。按3个head分组限制工作显存。block checkpoint减少激活驻留。对固定mask的Q/K/V/摘要梯度与独立显式softmax公式比较，所有输入路径通过检查；硬路由索引仍不可导，用候选换页差的sigmoid-margin STE训练路由系数。','','关键区别：native attention输出不detach，后续attention输入梯度没有被截断；路由特征/离散候选索引停止梯度是明确的STE近似。模型checkpoint分支只在训练事件启用，正常生成保持原始数值。','',f'早期和稳态检查均覆盖29层。早期smoke forward逐元素一致：{smoke["hard_forward_bit_exact"]}，29层梯度非零，反转最终输出误差后梯度方向近乎反向。']
    if cap:lines += [f'稳态338历史页：forward逐元素一致={cap["hard_forward_bit_exact"]}；首次反向{cap["backward_seconds"]:.3f}s，额外峰值显存{cap["extra_peak_bytes"]/2**30:.3f}GiB。这是共享GPU首次试测，不是稳定训练速度。']
    if cost:lines += ['',f'已提交事件的反向中位数{cost["median_backward_seconds"]:.3f}s，额外峰值显存最大{cost["max_extra_peak_gib"]:.3f}GiB；teacher＋梯度前向＋反向累计增加原有forward时间约{cost["extra_fraction"]:.1%}。不能沿用单层19%的成本估算。']
    lines+=['','## 训练协议','','8个train场景×2轮；每case chunk2/5、step0/3采4个事件。Adam lr=.003，参数限制在初始化±.1，梯度norm上限1。只用最终x0 NMSE，无attention局部辅助loss。4个val场景在固定初始化轨迹上比较初始/学习后输出，按逐事件误差比选epoch。回归视频不参与训练或选参。', '', '每case原子checkpoint保存模型和Adam状态，保留latest/previous；视频按完整单元恢复。目录上限1GiB、空闲下限50GiB，不落盘QKV。KV及输入辅助量5,096,033,584B；图工作区和训练激活另报。']
    if selection:
        lines+=['',f'验证选择epoch{selection["epoch"]}。','','| epoch | 初始x0 NMSE | 学习后x0 NMSE | 平均逐事件误差比 |','|---|---:|---:|---:|']
        for r in selection['rows']:lines.append(f'| {r["epoch"]} | {r["initial_nmse"]:.6f} | {r["trained_nmse"]:.6f} | {r["mean_ratio"]:.6f} |')
    lines+=['','## 前段闭环质量','','RGB[21,101)，seed1001；off为关闭修正，initial为原始全层gain路由，trained为全层输出梯度学习后路由。','','| 场景 | 配置 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---:|---:|---:|']
    for r in videos:
        q=r['quality'];lines.append(f'| {r["case"]} | {r["mode"]} | {q["lpips"]:.5f} | {q["ssim"]:.5f} | {q["psnr"]:.3f} |')
    lines+=['','梯度检查或验证loss下降不代表视频验收通过。只有trained在两个场景相对off和initial均降低LPIPS且不降低SSIM，才自动继续完整30秒训练后速度测试；否则报告质量未过关，不用不合格视频宣称加速收益。旧64.88%稳定方案仍是更高保真度参照。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(status=status,training_cases=len(train),gradient_updates=len(events),cost=cost,selection=selection,videos=videos))
if __name__=='__main__':main()
