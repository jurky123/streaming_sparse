"""Markdown evidence summary, never declares acceptance from local errors."""
import json
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_train_20261030')
def main():
    selection=json.loads((ROOT/'selection.json').read_text());val=selection['rows'][selection['selected_epoch']]['means']
    early=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance'/'early').glob('*/*.json'))]
    full=[json.loads(p.read_text()) for p in sorted((ROOT/'acceptance'/'full').glob('*/*.json'))]
    lines=['# 90% 稀疏率学习实验','','验证集选择 epoch2（第3轮），下面局部误差与闭环质量分别统计。','','| 模块 | 验证注意力 NMSE↓ | 相对 baseline 降幅 |','|---|---:|---:|']
    for mode,x in val.items():lines.append(f'| {mode} | {x:.6f} | {1-x/val["baseline"]:.2%} |')
    lines+=['','8个训练场景与4个验证场景；训练轨迹来自冻结原始global90路径，存在闭环分布迁移。swap相对joint的局部收益很小，不足以证明额外修正开销值得。','','## 前段闭环质量','','固定seed1001，未压缩RGB[21,101)。LPIPS越小越接近dense参考。旧64.88%补偿方案LPIPS为篮球0.25812、蝴蝶0.14104。90%方案必须同时检查与这些质量参照的差距。','','| 场景 | 模块 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---:|---:|---:|']
    for row in early:
        q=row['quality'];lines.append(f'| {row["case"]} | {row["mode"]} | {q["lpips"]:.5f} | {q["ssim"]:.5f} | {q["psnr"]:.3f} |')
    lines += ['',f'已完成 {len(early)}/10 个前段视频单元。','', '## 完整30秒速度','','包含生成、路由、维护与VAE；不含加载、预热、指标与编码。共享GPU测量需结合每条JSON中的telemetry判断，不能视为独占性能。','','| 场景 | 模块 | 生成+VAE秒 |','|---|---|---:|']
    for row in full:lines.append(f'| {row["case"]} | {row["mode"]} | {row["performance"]["total_generation_vae_seconds"]:.3f} |')
    lines+=['','常驻KV及辅助量5,095,914,800 B，低于5,096,079,360 B上限。稳态每tile精确41/410页，精确稀疏率90%；不包括摘要补偿、路由和dense路径。融合聚合6项数值检查通过；路由特征仍为PyTorch实现。','','局部NMSE下降不等于闭环视频质量提升；完整视频速度也不能由90%精确稀疏率直接推导。详见 [实验协议](PROTOCOL.zh.md)。']
    shifts=[json.loads(p.read_text()) for p in sorted((ROOT/'shift').glob('*.json'))]
    lines+=['','## 诊断与结论','','本轮10项前段消融中，学习组均未超过匹配baseline的LPIPS；也远未回到旧64.88%方案的保真度。不能验收为质量提升。真实闭环18点抽查中，融合V code与直接聚合BF16结果一致；另6项随机/环形回绕检查通过。恢复检查验证了24个训练单元、29层权重与优化器状态。','','| 学习后轨迹 | baseline局部NMSE | route | value | joint |','|---|---:|---:|---:|---:|']
    for r in shifts:
        m=r['means'];lines.append(f'| {r["case"]} | {m["baseline"]:.6f} | {m["route"]:.6f} | {m["value"]:.6f} | {m["joint"]:.6f} |')
    lines+=['','以上在joint自身轨迹上进行只读局部反事实比较，非独立生成的视频。route在两个场景均增大局部误差，提示训练源分布迁移；value仍减少局部误差，却不改善最终保真度，说明单层NMSE不是充分的闭环目标。训练只覆盖固定轨迹的少量query，不能把此结果外推成所有可学习聚合无效。','','## 稀疏率口径与后续方向','','按41chunks、30层、每chunk四次去噪加一次clean、各层相同缓存长度的dense对照估算，精确token对全流程削减约68.54%，稳态约69.48%。90%仅属于稀疏调用。摘要计算另计。保持clean和首层dense时，即使其他路径完全省掉，稳态也只有约77.20%的精确对削减上限；若目标是全流程80%–90%，必须改变这些dense路径。','','下一轮优先改变算法目标，而非继续扩大本轮线性路由训练：①混合dense/旧稳定/student轨迹训练，验证按轨迹来源分层；②路由学习“精确计算相对摘要补偿的边际收益”，而非只拟合页质量；③在训练场景加入短闭环去噪输出/latent目标和信赖域，防止局部下降但轨迹漂移；④预算按layer/step/query分配并严格累计总token对，单独研究clean路径，避免只报局部90%。单次swap应先证明稳定收益再支付额外开销。当前没有依据宣称理论稀疏计算不可能，但本轮部署尚未兑现质量目标。','','## 对比视频','','每格相同帧21–100，未做调色。左上旧64.88%补偿，右上90%匹配baseline，左下90%学习V，右下90%联合学习。','','- [篮球](comparisons/basketball_court_early90.mp4)','- [蝴蝶](comparisons/butterfly_garden_early90.mp4)']
    lines+=['','## 路由开销诊断','','9chunk插桩CUDA事件时间，包含当前摘要维护与路由PyTorch操作；不含VAE，非完整性能验收。共享GPU和插桩影响均存在，数值不能视作kernel独占延迟。','','| 模块 | 路由ms/次 | 精确注意力+补偿ms/次 |','|---|---:|---:|']
    for p in sorted((ROOT/'profile').glob('*.json')):
        r=json.loads(p.read_text());t=r['profile_gpu_ms'];n=r['sparse_calls'];lines.append(f'| {p.stem} | {t["route"]/n:.3f} | {t["sparse_attention"]/n:.3f} |')
    lines+=['','当前路由耗时已经超过精确稀疏attention+tail；高稀疏度减少了精确计算，却没有同比减少全页探针打分、特征归约、topk和临时张量。后续应融合logsumexp/特征/线性修正，减少完整概率及特征张量落地；但应先确定能保住闭环质量的路由目标，再优化其实现。理论上较少乘加量尚未转化成相应实测路由延迟。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(validation=selection,early=early,full=full))
if __name__=='__main__':main()
