"""Document IO analysis and bounded microbenchmarks without promoting a kernel."""
import hashlib
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parents[2]
    root = base/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024'
    out = root/'optimization3'
    read = lambda p: json.loads(p.read_text())
    source_seal = read(root/'optimization2/manifest.json')
    guarded_seal = read(root/'optimization2/guarded_manifest.json')
    old = {**source_seal['code'], **source_seal['prior_code'], **source_seal['original_code'],
           **guarded_seal['code'], **read(root/'e2e_manifest.json')['frozen_files']}
    for filename, digest in old.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    probes = {name: read(out/file) for name,file in [('parallel', 'tiled_probe.json'), ('recompute','reload_probe.json')]}
    rows = []
    for name, probe in probes.items():
        for setting in probe['settings']:
            rows.append(dict(experiment=name, **setting,
                score_plus_finish_ms=probe['score_only_ms']+setting['finish_ms'],
                paired_reference_ms=probe['reference_score_ms']+probe['reference_finish_ms']))
    assert len(rows) == 14 and all(r['changed_queries']==0 and r['candidates_max']==0 for r in rows)
    identity = read(out/'algebra_check.json')
    summary = dict(status='analysis_and_microbenchmarks_complete', adopted=False,
        selected_production_adapter='migration.unified_router.opt2_guarded.load_guarded',
        old_frozen_files_verified=len(old), tested_unique_head_queries=6144, configurations=14,
        timing_scope='30 warm eager component calls, same L24 H14040 first512 queries; excludes mask/candidate selection in both paths, includes action head; not CUDA Graph or E2E',
        numerical_scope='No final mask changes on this single block subset; no new video or full-prompt validation',
        algebra=identity, rows=rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# 路由的物化开销与 FlashAttention 式重构', '',
        '**判断：全页 content 的物化是重要热点，但不是唯一原因。路由仍扫描全部页，使用 FP32 计算；直接减少中间张量不能保证加速。当前正式入口仍为 Opt2 Guarded，本轮原型未替换它。**', '',
        '## 现有数据流和量级', '',
        '真实代表层：layer24、第 4 chunk、step3；12 heads、4680 queries、296 页、2220 个原型。以下为一次整层调用累计张量大小，GB/MB 使用十进制；执行按 512 queries 分批，不是同时占用的峰值显存，也不是硬件计数器实测 DRAM 流量。', '',
        '|中间量|大小|用途|', '|---|---:|---|',
        '|每原型 logits|498.7 MB|页内 softmax / logsumexp|',
        '|每页 scalar score|66.5 MB|阈值与候选选页|',
        '|每页 32 维 content|2127.8 MB|集合均值、离散程度、候选向量|',
        '|最多 16 个候选向量|115.0 MB|81 个动作的几何特征|',
        '|每 query 80-float 统计缓冲|18.0 MB|MLP 所需集合特征|', '',
        '完整 content 至少有一次全量写入和统计阶段的一次全量读取，约 4.26 GB 的逻辑访问，尚未算候选读取。当前已没有旧版大 cat；再优化的目标是 content 本体。最后的候选和统计合计约 133 MB，仅为完整 content 的 1/16，但并行 partial reduction 还需要额外缓冲。', '',
        '最终 Opt2 profile：路由 10.64 ms，attention 2.96 ms；其中 content 2.64 ms、集合统计 1.96 ms、动作 head 2.49 ms。统计 kernel 约 217–218 registers/thread，动作 head 150，说明片上资源压力也很明显。未测 Nsight Compute DRAM/SM 利用率，因此不能声称某个精确比例属于带宽瓶颈。', '',
        '15% 保留率主要减少后面的 attention；当前路由的 score、全体集合统计仍需要全部页。仅 score/content 主乘法约 24 GFLOP，还是 FP32 路径。A100 的 FP32 峰值约 19.5 TFLOPS，而 BF16 Tensor Core（不含结构化稀疏翻倍）约 312 TFLOPS；相同 FLOP 数不能按同一吞吐量估算。这些峰值也不是实际执行速度。来源：[NVIDIA A100 数据表](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf)。', '',
        '## 能借鉴 FlashAttention 的部分', '',
        'FlashAttention 的关键是让中间结果在片上被消费，分块维护小累加器，并在合适位置用重算换取更少显存访问。它并不是只减少乘法次数。来源：[原论文](https://arxiv.org/abs/2205.14135)、[Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。', '',
        '这里有一个额外依赖：初始 mask 取决于所有页 score 的均值和标准差；候选也依赖全页结果。因此更合理的是两阶段，而非把所有计算强塞进单次扫描。', '',
        '```mermaid', 'flowchart LR', '    A[query与缓存原型] --> B[第一阶段：每页scalar score]',
        '    B --> C[全页阈值与候选索引]', '    C --> D[第二阶段：小块content在片上计算]',
        '    D --> E[立即累加全体与已选集合统计]', '    D --> F[仅留下最多16个候选向量]',
        '    E --> G[原动作MLP与逐query mask]', '    F --> G', '```', '',
        '保留便宜的 score 缓冲，优先移除贵 32 倍的 content。第一版可以保留 logits，避免重复 FP32 QK GEMM；以后再判断 logits 重算是否划算。每个 query 的集合统计与候选仍独立，不引入 Q64 式共享选页。', '',
        '## 本轮原型实测', '',
        '实测同一真实 block 的前 512 queries × 12 heads，共 6144 个唯一 head-query。每项预热后测量 30 次 eager 组件调用；两边均不计 mask/候选筛选，均计 score 与最终动作 head。它不是完整路由 CUDA Graph 或端到端视频测速。', '',
        '|实现|score + 后处理 / ms|配对 Opt2 / ms|替代 content 的 partial+候选缓冲 / MB|',
        '|---|---:|---:|---:|']
    for name, label in [('parallel', '并行页块归约：最快配置'), ('recompute','重算页内权重：最快配置')]:
        row = min((r for r in rows if r['experiment']==name),key=lambda r:r['score_plus_finish_ms'])
        lines.append(f"|{label}|{row['score_plus_finish_ms']:.3f}|{row['paired_reference_ms']:.3f}|{row['partial_and_candidate_bytes']/1e6:.1f}|")
    lines += ['',
        '参考完整 content 为 232.8 MB / 512-query 批。并行页块最快配置将这一部分缩到 46.0 MB；重算方案最快配置为 79.4 MB。最省缓冲的重算配置为 32.6 MB，但更慢。此表不包含两边共同的 score、raw logits、统计输出等，不是总峰值显存。', '',
        '14 个配置均未改变这 6144 个 head-query 的最终 mask；候选向量逐 bit 一致。但集合中心最多约 1.4e-6、离散程度约 4.6e-5 的浮点差异，不能据此宣称视频无损。没有进行新版本端到端视频回归，因为这些实现尚无速度收益。', '',
        '省显存却变慢的实测结果说明还要控制布局转换、寄存器活跃范围、片上归约与算力利用率。页组串行流式版、这次并行展开版均不值得直接采用。重算在本次实现里减少了片上提取/交换开销，但仍未胜过 Opt2。', '',
        '环境为 Triton 3.2.0；直接 3D IEEE batched-dot 的初版触发编译器断言，当前原型改为显式 FP32 FMA。这个实现限制意味着负结果不能代表所有 IO-aware 路由算法都更慢。', '',
        '## 值得进一步验证的代数重构', '',
        '设一页的 P×32 投影原型为 V_b，页内归一化权重为 w_b。原 content 为 c_b = w_b V_b。后续真正需要的是全体/已选集合的加权均值与平方范数、候选 c_b；不要求所有 c_b 常驻显存。', '',
        '均值可以改写为按原型展开后的大矩阵乘法，而不输出每页向量：', '',
        '`μ_S = (Σ_b∈S t_b w_b V_b) / (Σ_b∈S t_b)`', '',
        '页内平方范数可借助预先缓存的小 Gram 矩阵：', '',
        '`G_b = V_b V_bᵀ`，`‖c_b‖² = w_b G_b w_bᵀ`', '',
        '`variance_S = (Σ_b∈S t_b ‖c_b‖²) / (Σ_b∈S t_b) − ‖μ_S‖²`', '',
        '历史 G_b 可以随原型缓存复用，当前 P16 的 G_b 也仅 16×16；候选页再单独计算 32 维向量。对真实 72 queries×12 heads 的 FP64 代数检查通过，见 algebra_check.json。', '',
        '这是待优化内核设计，不是已经获得加速的实现：双集合均值和 Gram 二次型可能增加乘法总量，能否由较好的矩阵形状和更少 IO 抵消必须测速。FP32 中 E[x²]−E[x]² 可能发生消减，特别是单页集合；可用分块加权 Welford 合并或两遍中心化归约保证稳定性，但归约顺序仍会变。', '',
        '采用门槛：先在三种历史长度与 K32/V32 上取得真实完整路由收益，再接入原决策边界保护并跑 16 组完整视频回归。旧保护内核依赖完整 content，不能原封不动保留，否则省 IO 的收益消失；应仅为边界 query 重算参考特征。仍保持逐 query、FP32 路由、P16/P8/P4/P2 和原保留预算。', '',
        '## 代码与数据', '',
        f'- [分块与重算原型]({base}/migration/unified_router/opt3_tiled.py)',
        f'- [代数恒等式验证]({base}/migration/unified_router/opt3_identity.py)',
        '- [并行配置实测](tiled_probe.json)、[重算配置实测](reload_probe.json)、[代数误差](algebra_check.json)、[汇总](summary.json)。',
        '- [仍推荐的 Opt2 报告](../optimization2/REPORT.md)。原代码、权重和冻结配置 156 份哈希均通过校验。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    files = list(Path(__file__).parent.glob('opt3_*.py'))
    (out/'manifest.json').write_text(json.dumps(dict(code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        reference='Opt2 Guarded', old_files_unchanged=len(old), production_changed=False),indent=2)+'\n')
    print(json.dumps(dict(configurations=len(rows), unique_head_queries=6144, frozen_files_verified=len(old),adopted=False),indent=2))


if __name__ == '__main__':
    main()
