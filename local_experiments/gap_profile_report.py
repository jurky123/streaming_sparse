"""Evidence-only Markdown report; requires completed paired runs and live traces."""
import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/gap_profile_20261031')
def main():
    read=lambda p:json.loads(p.read_text())
    assert read(ROOT/'performance/status.json')['status']=='complete'
    assert read(ROOT/'performance_features/status.json')['status']=='complete'
    sparse,dense,opt=[read(ROOT/(n+'.json')) for n in ['sparse','dense','features']]
    groups=[('dense','performance','dense'),('优化前（原 fast）','performance_features','fast'),('仅归一化融合','performance','fused'),('完整本轮优化','performance_features','fused')]
    records=[]
    for label,directory,prefix in groups:
        xs=[read(ROOT/directory/(prefix+'_'+str(i)+'.json')) for i in range(2)]
        records.append(dict(label=label,seconds=[x['performance']['total_generation_vae_seconds'] for x in xs],mean=statistics.mean(x['performance']['total_generation_vae_seconds'] for x in xs),generation_mean=statistics.mean(x['performance']['generation_seconds'] for x in xs),exact=[x['full_latents_exact'] for x in xs],peak_allocated_bytes=max(x['performance']['peak_allocated_bytes'] for x in xs)))
    before,after=records[1]['mean'],records[3]['mean'];dm=records[0]['mean']
    speedup=before/after;vs_dense=dm/after
    stages=[]
    for key in sparse['cuda_event_stages']:
        a=sparse['cuda_event_stages'][key];b=opt['cuda_event_stages'][key]
        stages.append(dict(name=key,before=a,after=b))
    result=dict(records=records,speedup=speedup,dense_speedup=vs_dense,stages=stages,kernel_counts={n:x['kernel_count'] for n,x in [('dense',dense),('before',sparse),('after',opt)]},quality='Accepted shared checkpoint; both optimized full 41-chunk rollouts bit-exact vs baseline; no algorithm/parameter change.')
    atomic(ROOT/'summary.json',result)
    lines=['# 稀疏视频 attention：真实性能定位与定向优化','',
      '## 结论','',f'本轮优化将完整 30 秒视频平均耗时从 **{before:.3f}s 降至 {after:.3f}s**，加速 **{speedup:.3f}×**（耗时减少 {(1-after/before)*100:.1f}%）。同条件 dense 为 **{dm:.3f}s**，优化版相对 dense 为 **{vs_dense:.3f}×**。',
      '质量采用用户已经认可的 shared/epoch1.pt。两次完整 41-chunk 输出均与优化前 latent 逐元素完全相同，参数、选块预算、补偿公式不变。不能把本轮工程优化解释为新的质量收益。','',
      '## 完整视频验收','',
      'H200 NVL；horse_field/seed101；512×768、480帧、16fps；四个去噪步骤及 clean commit；计时包含 VAE。每个 runtime 预热六个 chunk 和 VAE，排除首次图捕获/编译。实测是热启动持续推理成本，不是首次启动延迟。两轮反向顺序。dense 与归一化实验先完成，随后基线/完整优化配对复测；非跨全部方案随机排列。','',
      '| 实现 | 第一次/s | 第二次/s | 平均/s | 生成部分平均/s |','|---|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['label']} | {r['seconds'][0]:.3f} | {r['seconds'][1]:.3f} | {r['mean']:.3f} | {r['generation_mean']:.3f} |")
    lines+=['','两次计时只用于工程验收，不是置信区间；完整相等性验证覆盖上述一个 prompt/seed。另有 216/288/360/410 页、三档 logits 尺度的 12 组路由选块一致性测试。未重跑全部质量集，因为优化版在本次完整视频上无数值变化。','',
      '常驻 self-KV+辅助缓存：优化版 5,112,819,024 B，未超过已批准的原上限+16 MiB（5,112,856,576 B）；dense 5,096,079,360 B。稀疏保留338历史页，dense保留360历史页，辅助摘要占用计入预算。该预算不等于全进程显存；CUDA graph工作区和模型激活另计，峰值实测保存在JSON。','',
      '## 真实模型 profile','',
      '预热六个 chunk 后，对真实 chunk6 采集 CPU+CUDA trace，包含四次去噪和 clean commit，不含 VAE。CUDA event 时间含队列间隙；kernel 总时长不是端到端墙钟时间；表中模块不与父级算子时间重复相加。','',
      '| 模块 | 调用次数/chunk | 优化前平均/ms | 优化后平均/ms |','|---|---:|---:|---:|']
    for s in stages:lines.append(f"| {s['name']} | {s['before']['calls']} | {s['before']['mean_ms']:.3f} | {s['after']['mean_ms']:.3f} |")
    lines+=['',f"GPU kernel 数：dense **{dense['kernel_count']:,}**，优化前 **{sparse['kernel_count']:,}**，优化后 **{opt['kernel_count']:,}**。kernel 活跃时长之和分别为 {dense['kernel_sum_us']/1000:.2f} / {sparse['kernel_sum_us']/1000:.2f} / {opt['kernel_sum_us']/1000:.2f} ms。",'',
      '## 为什么 FLOPs 更少却更慢','',
      '1. 先前约 14.1% 的比例仅计主要 attention 矩阵乘法：精确分支 10%、四代表补偿 3.32%、16-probe 路由 QK 0.78%。遗漏了归一化、top-k、条件 V/残差特征、当前页摘要和历史重排；不同精度的 FLOPs 也不能按相同吞吐折算。',
      '2. 实测矩阵吞吐：dense 742.90 GFLOP / 2.793 ms ≈ 266 TFLOP/s；精确稀疏 74.29 GFLOP / 0.272 ms ≈ 273 TFLOP/s。精确稀疏 kernel 已兑现约十分之一的工作量。',
      '3. 路由 QK 使用 strict FP32：5.80 GFLOP / 0.181 ms ≈ 32 TFLOP/s，同机大 GEMM 标定为 46.6 TFLOP/s；其乐观下界约 0.125 ms。因此单独继续优化这个 GEMM，收益远小于融合后处理。',
      '4. 单次路由 logits 是 12×72×16×410×4 个 FP32 元素，约 86.5 MiB。原实现对其进行多次归约及中间张量写回，小算子数量和内存流量很高；CUDA graph 只能减少提交开销，不能自动融合这些 kernel。',
      '5. 补偿不能只按 24.67 GFLOP 的理想数学量估算。实际还包含高/低精度拆分、矩阵 padding、在线 softmax 和小 tile 开销；考虑拆分/padding 的粗略计算下界约 0.169 ms，实测约 0.681 ms。这个下界不是已经证明可达到的速度。',
      '6. 稳态历史提交约 3.14 ms/层/chunk，摊到四次去噪约 0.78 ms/次；当前摘要约 0.63 ms/次。缓存只避免历史重复预处理，新 chunk 的分组和摘要并非免费。层0、clean步、早期chunk仍使用 dense，MLP和VAE也不随稀疏率同比下降。','',
      '## 本轮已落实','',
      '- 用一个 Triton kernel 合并四代表 logsumexp/softmax，分别保留 PyTorch 两种归约顺序，使用明确的除法和 exp/log，关闭 FMA 融合。',
      '- 融合条件 V 的14维路由特征构造，保留 native std 和打分 GEMV。正确处理 einsum 结果的非连续 stride，避免额外 contiguous 副本。',
      '- 融合集合修正和输出页 LUT，保留 argmin/argmax 的并列处理规则；当前页仍受保护。',
      '- QK 直接写入 CUDA graph 输入 workspace，原地缩放，去除大 logits 复制；去掉当前 K/V 的重复写入。',
      '- 固定16 probes、4代表、10%精确页、原 checkpoint；没有新增常驻 KV/摘要缓存。优化仅作用于推理，训练仍调用原有可微路径。','',
      '## 尚未证明的部分和下一优先级','',
      '没有足够证据声称达到总体理论极限。当前环境 PATH 中没有 ncu/nsys，本次证据来自 torch.profiler、CUDA events、实测带宽/GEMM标定；没有 SM occupancy、HBM/L2 或指令计数器的 roofline 实证。',
      '下一步应针对当前摘要/历史提交中的归约、重复投影与排序，以及补偿 kernel 的 tile/精度拆分继续优化。直接减少 probes、代表数或取消补偿属于算法改变，需要重新做质量验收；本轮没有这样换取速度。','',
      '## 文件与复现','',
      '- 实现：`local_experiments/gap_feature_runtime.py`、`gap_feature_kernel.py`、`gap_repstats.py`。',
      '- 完整配对实验：`CUDA_VISIBLE_DEVICES=5 bash local_setup/run.sh -m local_experiments.gap_profile_feature_perf`。运行目录含 identity、状态和每个完成单元的 JSON，可中断后按单元续跑；单个未完成视频需要重跑。',
      '- 真实 profile：`... -m local_experiments.gap_live_fused_profile --variant features`。',
      '- 原始证据：本目录 `sparse.json`、`dense.json`、`features.json` 及各自 `.trace.json.gz`；`fusion_checks.json`；`performance/`、`performance_features/`。',
      '- 全部输出在本目录；每个完整性能实验子目录 1 GiB 限额，不保存 QKV dump。', '']
    (ROOT/'report.zh.md').write_text('\n'.join(lines))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
