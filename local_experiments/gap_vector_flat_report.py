"""Read-only evidence aggregation: speed first, two-case quality smoke only."""
import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/gap_vector_flat_20261031')
def main():
    read=lambda p:json.loads(p.read_text())
    assert read(ROOT/'vector_perf/status.json')['status']=='complete'
    assert read(ROOT/'flat_perf/status.json')['status']=='complete'
    q=read(ROOT/'quality/summary.json');rows=[]
    for label,folder,prefix,count in [('第一阶段dense','vector_perf','dense',2),('上一轮已验证版本','vector_perf','before',3),('向量化归一化','vector_perf','after',3),('本轮dense','flat_perf','dense',2),('向量化路由基线','flat_perf','before',2),('代表级补偿','flat_perf','after',2),('代表级补偿+权重归约融合','flat_perf','ordered',2)]:
        results=[read(ROOT/folder/f'{prefix}_{i}.json') for i in range(count)]
        seconds=[x['performance']['total_generation_vae_seconds'] for x in results]
        rows.append(dict(label=label,seconds=seconds,mean=statistics.mean(seconds),median=statistics.median(seconds),generation_mean=statistics.mean(x['performance']['generation_seconds'] for x in results),all_accepted_latents_exact=all(x['full_latents_exact'] for x in results),flat_branch_latents_exact=[x.get('flat_branch_latents_exact') for x in results]))
    baseline,dense,latest=rows[4]['mean'],rows[3]['mean'],rows[6]['mean']
    profile=read(ROOT/'profile/features.json') if (ROOT/'profile/features.json').exists() else None
    tuning=read(ROOT/'flat_tuning.json') if (ROOT/'flat_tuning.json').exists() else None
    atomic(ROOT/'summary.json',dict(rows=rows,quality=q,vs_vector_speedup=baseline/latest,vs_dense_speedup=dense/latest,theoretical_limit_demonstrated=False,profile_present=profile is not None))
    lines=['# 路由与补偿优化验证','',
      f'本轮最终组合平均 **{latest:.3f}s**，同轮向量化路由基线 **{baseline:.3f}s**，耗时下降 **{(1-latest/baseline)*100:.2f}%**；dense **{dense:.3f}s**，相对dense加速 **{dense/latest:.4f}×**。接近dense不等于已经达到理论速度，极小差异不作稳健加速结论。','',
      '## 完整视频速度','',
      '同一H200、同模型和种子horse_field/101，512×768、480帧/30秒，包含VAE。稀疏率90%、16 probes、4代表、物理历史338页及原辅助预算不变，无rebase。初始化分别预热六个chunk和VAE，然后交替切换已预热实例，避免每次重建。切换稀疏实例共享同一原始KV缓存，保留各自图输入/模型；每次运行重置历史状态。首次编译/图捕获不计入热启动时间。计时前后检查同卡无其他GPU进程。','',
      '| 实现 | 各次完整耗时/s | 平均/s | 生成部分平均/s |','|---|---|---:|---:|']
    for r in rows:lines.append(f"| {r['label']} | {', '.join(f'{x:.3f}' for x in r['seconds'])} | {r['mean']:.3f} | {r['generation_mean']:.3f} |")
    lines+=['','两个阶段各自配对，不把跨阶段单次最快值拼成加速比。各次结果全部列出；第一轮基线偏慢会影响均值，不能仅取最大收益。并存的非活跃runtime工作区计入记录中的显存峰值，因此该峰值不是单实例部署峰值。','',
      '## 已确认的实现优化','',
      '1. **归一化访存向量化**：同样的四代表logsumexp/softmax运算，由四个跨步加载改为连续向量加载。局部约0.1275→0.0525ms；四种页数测试完全一致，三次完整41chunk latent也与已验收版本逐元素一致。',
      '2. **权重乘法+mean16融合**：对照PyTorch非连续维归约的四个串行累加器及组合顺序，保留原算术顺序、关闭FMA收缩；不再物化巨大的prob×within临时张量。局部约0.1149→0.0525ms；216/288/360/410页、多个block参数均逐元素一致。两次完整视频在加此融合前后也完全一致（参照同一代表级补偿版本）。',
      '3. **代表级补偿**：页内base+3个Haar细节先解码成4个rank8 V代表，直接做代表级online softmax，保持每代表16 token质量、选中页排除规则、共同分母及最终低秩解码。数学等价，但浮点求和顺序改变，视频不逐元素一致。局部tail+merge约0.6938→0.5348ms；不是通过降低probe数、代表数、稀疏预算或删除补偿换速度。',
      '预解码代表是临时workspace，不新增常驻KV；常驻self-KV+辅助缓存仍约5,112,819,024B，在已批准原上限+16MiB之内。','',
      '## 按用户要求的快速质量检查','',
      '只检查两个原有样本的前段RGB第21–100帧，未跑完整质量集。重新解码同一dense和已验收shared checkpoint的已有latent，用同一实现重算指标。LPIPS越低越好，SSIM/PSNR越高越好；下表是新补偿减已验收版本、两者都相对dense的质量差。','',
      '| 样本 | ΔLPIPS | ΔSSIM | ΔPSNR/dB |','|---|---:|---:|---:|']
    for r in q['rows']:lines.append(f"| {r['case']} | {r['delta']['lpips']:+.6f} | {r['delta']['ssim']:+.6f} | {r['delta']['psnr']:+.4f} |")
    d=q['mean_delta'];lines.append(f"| 平均 | {d['lpips']:+.6f} | {d['ssim']:+.6f} | {d['psnr']:+.4f} |")
    lines+=['','快速筛查通过：两个样本没有一致退化，平均指标略好；交通LPIPS变差，鸟类改善。样本太少，不能据此宣称整体质量提高或统计非劣。',
      '数值误差可经自回归和硬路由累积，不能因为合成attention NMSE约1e-9就宣称视频相同。两样本新旧视频之间LPIPS约0.193/0.101，确有可见差异；验收依据是上述与dense的前段质量比较。','',
      '## 真实profile与剩余热点','']
    if profile:
        lines+=['预热六个chunk后profile真实chunk6（四次去噪+clean，不含VAE）。事件时间含队列间隙；不与父级耗时重复相加。','', '| 模块 | 每次平均/ms | 调用/chunk |','|---|---:|---:|']
        for name,x in profile['cuda_event_stages'].items():lines.append(f"| {name} | {x['mean_ms']:.4f} | {x['calls']} |")
        lines+=['',f"GPU kernel数量 {profile['kernel_count']:,}，kernel活跃时间总和 {profile['kernel_sum_us']/1000:.2f}ms；这不是含VAE的端到端耗时。"]
    else:lines+=['真实profile尚未完成，不能将微基准直接替代真实模块耗时。']
    lines+=['','仍不能宣称接近总体理论极限。原来的约14%是主要attention矩阵乘法比例，未完整包含归约、重排、摘要、softmax、精度拆分/padding，以及保留的dense调用和骨干/VAE开销。',
      '新补偿减少了嵌套归一化及多分量处理，实际AV还需处理FP32代表的高低位，算术量并非同比下降。因此“FLOPs更少”不能单独预测kernel速度。当前账户的Nsight GPU硬件计数器权限仍不可用，未取得SM/HBM/L2 roofline实证。']
    if tuning:
        lines+=['','## 后续补偿tiling探索','', '以下只作为局部资源/速度证据，尚未据此替换上述完整验收配置（BQ64/BR64/4warps/1stage）。','', '| BQ/BR/warps/stages | ms | 寄存器/线程 | 共享内存/CTA | 与flat基准完全一致 |','|---|---:|---:|---:|---|']
        for x in tuning['rows']:
            if 'ms' in x and 'invalid_reason' not in x:lines.append(f"| {x['bq']}/{x['br']}/{x['warps']}/{x['stages']} | {x['ms']:.4f} | {x['registers']} | {x['shared_bytes']} | {x['exact']} |")
    lines+=['','BQ128候选跨越两个独立的64-query选页掩码，不符合当前kernel的掩码语义，已明确排除；并非可接受的浮点误差。','', '## 代码与结果','',
      '- 数值不变路径：`gap_vector_runtime.py`、`gap_ordered_runtime.py:GapOrderedRuntime`。',
      '- 补偿组合候选：`gap_ordered_runtime.py:GapFlatOrderedRuntime`，本轮完整计时及快速质量检查对象。训练可微路径仍使用原实现。',
      '- 完整速度：`vector_perf/`、`flat_perf/`；快速质量：`quality/`，含两个新视频及原视频路径。',
      '- profile：`profile/features.json`及压缩trace。不是HTML报告。',
      '- 每次实验有身份校验、运行锁与逐单元落盘；未完成单元可重跑，磁盘剩余至少50GiB、单实验子目录上限1GiB。','']
    (ROOT/'report.zh.md').write_text('\n'.join(lines));print(json.dumps(dict(vs_vector=baseline/latest,vs_dense=dense/latest,seconds=latest,quality_screen=q['passed_screen'])))
if __name__=='__main__':main()
