"""Report the selected per-query adapter; verify sealed inputs before publishing."""
import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    root = Path(parser.parse_args().directory).resolve()
    out = root / 'optimization2'
    seals = read(out / 'manifest.json')
    guarded = read(out / 'guarded_manifest.json')
    frozen = read(root / 'e2e_manifest.json')['frozen_files']
    digests = {**seals['code'], **seals['prior_code'], **seals['original_code'],
               **guarded['code'], **frozen}
    for path, expected in digests.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

    speed, samples = {}, []
    folders = {'Dense': 'benchmark/Dense', 'Fast1-K32': 'benchmark/Fast1-K32',
               'Fast1-V32': 'benchmark/Fast1-V32', 'Guarded-K32': 'guarded_K32',
               'Guarded-V32': 'guarded_V32'}
    hashes, gpus = {}, {}
    for name, folder in folders.items():
        records = read(out / folder / 'timings.json')['records']
        rows = records[1:]
        assert len(rows) == 3 and len({r['latent_sha256'] for r in records}) == 1
        hashes[name] = rows[0]['latent_sha256']
        gpus[name] = rows[0]['gpu'].split(',')[1].strip()
        speed[name] = dict(variant=name, n=3,
            generation_s=statistics.mean(r['generation_seconds'] for r in rows),
            generation_std_s=statistics.stdev(r['generation_seconds'] for r in rows),
            decode_s=statistics.mean(r['decode_seconds'] for r in rows),
            total_s=statistics.mean(r['total_seconds'] for r in rows),
            peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in rows))
        samples.extend(dict(variant=name, **r) for r in rows)
    assert len({r['first_chunk_sha256'] for r in samples}) == 1
    for kind in ['K32', 'V32']:
        assert hashes['Guarded-' + kind] == hashes['Fast1-' + kind]
        assert gpus['Guarded-' + kind] == gpus['Fast1-' + kind]
    quality = []
    for rank in range(8):
        assert read(out / f'guarded_validation/rank{rank}.json')['passed']
        quality.extend(read(out / f'guarded_validation/results_rank{rank}.json'))
    assert len(quality) == len({(r['prompt_id'], r['mode']) for r in quality}) == 16
    assert all(r['latent_bit_identical'] and r['rgb_bit_identical'] for r in quality)
    assert read(out / 'profile_guarded/modules.json')['observer_latents_bit_identical']
    profiles = dict(Fast1=read(root / 'optimization/profile_q1/kernel_attribution.json'),
                    Guarded=read(out / 'profile_guarded/kernel_attribution.json'))
    module_maps = {k: {r['module']: r['kernel_ms'] for r in v['modules']} for k, v in profiles.items()}
    kernel_maps = {}
    for name, profile in profiles.items():
        totals = collections.defaultdict(float)
        for row in profile['kernels']:
            totals[row['kernel']] += row['milliseconds']
        kernel_maps[name] = dict(totals)
    prior_quality = read(root / 'optimization/summary.json')['quality']
    rate = read(out / 'guard_rate.json')
    assert rate['all_masks_bit_identical']
    summary = dict(selected_adapter='migration.unified_router.opt2_guarded.load_guarded',
        query_stride=1, tf32=False, speed=speed, validation=quality,
        quality_scope='8 Val prompts x K32/V32, seed0; bitwise equal to Fast1, not Dense',
        quality_vs_dense={k: prior_quality['Fast-' + k] for k in ['K32', 'V32']},
        profile_modules=module_maps, profile_kernel_totals_ms={k: v['total_kernel_ms'] for k,v in profiles.items()},
        representative_guard=rate, sealed_files_verified=len(digests))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(out / 'speed_summary.csv', list(speed.values()))
    write_csv(out / 'timing_samples.csv', samples)
    write_csv(out / 'quality_bitwise_checks.csv', quality)
    lines = ['# SlimAV v10 第二轮热点优化：保留逐 query 路由', '',
        '**最终采用 `opt2_guarded.load_guarded`：K32/V32 生成耗时约 7.91/7.99 秒，相比上一轮逐 query 版减少约 27%。8 个 Val prompt × 两组 head 的完整 latent 与解码 RGB 均逐 bit 一致。**', '',
        'Q64 不作为可用方案。本轮没有共享 query 的选页结果，没有重训、换 checkpoint 或降低预算。前期无保护 Opt2 和 strict 分支仅保留为消融证据，不是推荐入口。', '',
        '## 端到端计时', '',
        '单张 A100-SXM4-40GB；45 帧、480×832、4 chunks × 4 去噪步；相同 Val prompt 与 seed0。每项一次完整预热后计时三次。K32 前后均在 GPU1，V32 前后均在 GPU2；基线和优化计时分阶段进行，质量检查及 profiler 在计时后运行。', '',
        '生成包含原流程的上下文重建；总计另含 VAE 与 RGB 回传。不含加载、编译、文本编码、质量指标及视频文件编码。', '',
        '|方案|生成 / s|VAE及回传 / s|总计 / s|生成标准差 / s|',
        '|---|---:|---:|---:|---:|']
    for name, r in speed.items():
        lines.append(f"|{name}|{r['generation_s']:.3f}|{r['decode_s']:.3f}|{r['total_s']:.3f}|{r['generation_std_s']:.4f}|")
    for kind in ['K32', 'V32']:
        a, b = speed['Fast1-' + kind], speed['Guarded-' + kind]
        lines.append(f"\n{kind}：生成减少 {1-b['generation_s']/a['generation_s']:.2%}（{a['generation_s']/b['generation_s']:.3f}×），含 VAE 总时长减少 {1-b['total_s']/a['total_s']:.2%}。")
    lines += ['', '仍慢于 Dense；15% 逻辑保留率不能直接换算成端到端加速。', '',
        '## 热点和实现', '',
        '代表层为第 4 chunk、step3、layer24，4680 queries、18720 KV tokens。以下是 GPU kernel 累计耗时，不是 CPU scope 或墙钟时间。使用 CUDA runtime/driver correlation 将每个 kernel 唯一归到发起它的 scope；CUDA Graph 内节点归到 graph launch。所有 kernel 均完成归属，避免 profiler 对 graph/Triton 的 scope 漏计。', '',
        '|模块 / ms|上一轮 Fast1|最终 Guarded|', '|---|---:|---:|']
    for key, label in [('route_graph_copy_and_replay', '路由（含输入复制 kernel）'),
                       ('sparse_attention_including_union', '稀疏 attention 与页并集'),
                       ('current_prototypes', '当前原型'), ('prototype_projection', '原型投影'),
                       ('ffn', 'FFN'), ('cross_attention', '交叉注意力')]:
        lines.append(f"|{label}|{module_maps['Fast1'][key]:.3f}|{module_maps['Guarded'][key]:.3f}|")
    lines.append(f"|整层 kernel 合计|{profiles['Fast1']['total_kernel_ms']:.3f}|{profiles['Guarded']['total_kernel_ms']:.3f}|")
    lines += ['',
        '1. **直接写出 content。** 原版逐组 softmax、content BMM 后拼接，content BMM 与大 cat 在该层分别耗时约 5.04/3.10 ms。现在将每页 softmax、加权与目标布局写入融合，最终 `_content` 约 2.64 ms。避免额外 cat 对约 2.13 GB 逻辑张量的读写；仍分批生成完整 content，没有宣称消除了这个张量。',
        '2. **拆分集合统计与动作 MLP。** 原 `_finish` 约 7.95 ms；最终 `_statistics` 约 1.96 ms，`_guard_actions` 约 2.49 ms，另加少量边界重算。缩短不同阶段数据同时存活的时间。统计 kernel 仍有较高寄存器压力，不能理解成所有 kernel 的寄存器数都下降。',
        '3. **跳过不必要的候选填洞。** 候选槽已完整时跳过循环，保留原 tie-break、哈希和候选结果；候选 kernel 从约 1.94 ms 降至 0.82 ms。',
        '4. **调整 attention 调度。** 32-query kernel block、4 warps、3 stages；组内每个 query 仍应用各自 mask。这里的 32 只是 GPU 调度尺寸，不是 Q32 共享路由。', '',
        '## 数值保护与视频一致性', '',
        '优化仍使用 FP32 路由、BF16 主干，普通 TF32 关闭。score 融合内核开启 FP32 fused multiply-add，以匹配上一轮 Inductor 的运算约定；关闭 FMA 反而造成约 1e-6 级 score/content 差异。完整代表层的 FMA 对齐检查中，score 与 content 均逐 bit 一致。', '',
        '动作 utility 的第一、第二名差距 ≤0.001 或最佳值非有限时，GPU 上用原融合特征/MLP kernel 重算该 query。阈值作用于动作分数差距，不是保留率阈值；没有 CPU 同步。代表层 56160 个 head-query 中只重算 111 个（0.198%），最终 mask 与参考全同。该比例只代表此层。', '',
        '**完整验证：8 个已注册 Val prompt × K32/V32，seed0，共 16 次生成。全部 latent、全部未压缩解码 RGB 均与上一轮 Fast1 逐 bit 相同，latent NMSE=0、RGB 最大误差=0。** 计时的三次重复一致，首 chunk 与 Dense 一致，profile 观察器不改变输出。', '',
        f"因此这些已测样本对 Dense 的 LPIPS 保持原 Fast1 数值：K32 {prior_quality['Fast-K32']['lpips']:.8f}，V32 {prior_quality['Fast-V32']['lpips']:.8f}（仅后续稀疏帧）。这不是与 Dense 无损，也不是对任意新输入的数值等价证明。", '',
        '首 chunk 与 layer0 Dense；后续层逐 query 稀疏，首 chunk KV 按普通历史处理。当前 P16、历史随距离 P8/P4/P2；β、约 15% 预算、冻结 backbone/LoRA/head/basis/normalization 均不变。未访问 Test 或据本轮结果重新选择训练 checkpoint。', '',
        '## 保留的消融证据', '',
        '- 无保护 Opt2 更快，约 7.78/7.87 秒，但存在浮点决策漂移；V32 seal 场景 LPIPS 比 Fast1 高约 0.027，未采用。`quality/` 中的视频属于该分支，不能当作最终 Guarded 视频。',
        '- 仅保留原 score、调整拆分 kernel 的 strict 分支约 10.04/10.12 秒，16 组中仅 9 组 latent 完全一致，未采用。',
        '- 融合流式 FP64 矩统计的尝试更慢，未采用；动作 MLP 代数分解收益较小，也未采用。', '',
        '## 剩余瓶颈', '',
        f"最终代表层路由仍占 kernel 时间的 {module_maps['Guarded']['route_graph_copy_and_replay']/profiles['Guarded']['total_kernel_ms']:.1%}。score 的 FP32 GEMM、全页 content/统计与动作 head 是下一轮重点；直接去掉完整 content 的首版尝试尚未获益，需要重新设计归约和候选向量提取。",
        '页并集让实际 attention 工作量高于每 query 的逻辑保留率；首 chunk、layer0、上下文重建、FFN、投影和约 1.9 秒 VAE 也不随稀疏率下降。本轮没有使用共享 query 路由来换速度。', '',
        '## 代码、数据与复现', '',
        '- [最终源代码冻结记录](guarded_manifest.json)、[前期及原始代码冻结记录](manifest.json)。',
        '- [速度汇总](speed_summary.csv)、[逐次计时](timing_samples.csv)、[16 组一致性检查](quality_bitwise_checks.csv)、[机器可读汇总](summary.json)。',
        '- [最终模块 profile](profile_guarded/modules.csv)、[kernel 明细](profile_guarded/kernels.csv)、[Perfetto trace](profile_guarded/trace.json.gz)。',
        '- [FMA 检查](fma_probe.json)、[边界重算率](guard_rate.json)、[早期边界正确性检查](correctness.json)。',
        '- [速度与模块耗时图](speed_profile.png)。', '']
    code = Path(__file__).resolve().parent
    for f in ['opt2_guarded.py', 'opt2_guarded_kernels.py', 'opt2_precision.py',
              'opt2/score.py', 'opt2/features.py', 'opt2/candidates.py', 'opt2/attention.py']:
        lines.append(f'- [{f}]({code / f})')
    lines += ['', '```python', 'from migration.unified_router.opt2_guarded import load_guarded',
        'net, config = load_guarded(experiment_directory)', "net.set_mode('K32-Set')  # 或 V32-Set",
        '# 继续使用原 trajectory，置于 torch.no_grad() 和 sdpa_backend() 下。', '```', '',
        '在项目根目录复现计时（output 使用新目录）：', '', '```bash',
        'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 \\\nstreaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap \\\nmigration.unified_router.opt2_guarded \\',
        f'  --directory {root} \\', '  --output /tmp/slimav_opt2_new_benchmark --mode K32-Set', '```', '',
        '适配器限定当前实验的 batch1、12 heads、128 维、4680-token chunk 与 64-token 页；CUDA Graph 静态缓冲用于顺序推理。默认原实验入口未改，最终优化需显式使用 load_guarded。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    keys = list(speed)
    gen = [speed[k]['generation_s'] for k in keys]
    dec = [speed[k]['decode_s'] for k in keys]
    axes[0].barh(keys, gen, label='Generation', color='#427fa6')
    axes[0].barh(keys, dec, left=gen, label='VAE + transfer', color='#a2c8cf')
    for i, k in enumerate(keys):
        axes[0].text(gen[i] + dec[i] + .12, i, f'{gen[i]+dec[i]:.2f}s', va='center')
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 15)
    axes[0].set_xlabel('Seconds / 45-frame video'); axes[0].legend(fontsize=8)
    axes[0].set_title('Same per-query output on 16 Val cases')
    parts = [('Routing', 'route_graph_copy_and_replay'), ('Attention', 'sparse_attention_including_union')]
    left = [0., 0.]
    for label, key in parts:
        values = [module_maps[n][key] for n in ['Fast1', 'Guarded']]
        axes[1].barh(['Fast1', 'Guarded'], values, left=left, label=label)
        left = [a+b for a,b in zip(left, values)]
    values = [profiles[n]['total_kernel_ms']-left[i] for i,n in enumerate(['Fast1', 'Guarded'])]
    axes[1].barh(['Fast1', 'Guarded'], values, left=left, label='Other')
    axes[1].invert_yaxis(); axes[1].set_xlabel('Cumulative GPU kernel ms / representative layer')
    axes[1].set_title('Layer24, chunk4, step3'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'speed_profile.png', dpi=160)
    fig.savefig(out / 'speed_profile.pdf'); plt.close(fig)
    verification = dict(passed=True, sealed_files_verified=len(digests),
        complete_latent_and_rgb_bitwise_checks=16, clean_timed_samples=15,
        matched_gpu_per_mode=True, first_chunk_matches_dense=True,
        profiling_preserves_output=True, kernel_attribution_complete=True)
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
