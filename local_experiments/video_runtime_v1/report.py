"""Collect existing evidence only; never launches a generation or benchmark."""
import json,shutil
from pathlib import Path
from .weights import ROOT
from ..learned_cache.io import atomic,digest

def run():
    base=ROOT/'artifacts/video_runtime_v1';dest=ROOT/'docs/research/evidence/video_runtime_v1';dest.mkdir(parents=True,exist_ok=True)
    bench=json.loads((base/'benchmark/benchmark.json').read_text());video=json.loads((base/'video_benchmark/result.json').read_text());validation=json.loads((base/'validation/validation.json').read_text())
    videos=[]
    for name in ('pilot','long_pilot'):
        status=json.loads((base/name/'batch/status.json').read_text());assert status['stage']=='complete'
        for path in sorted((base/name/'videos').glob('*/*/*/generation.json')):
            gen=json.loads(path.read_text());post=json.loads((path.parent/'postprocess.json').read_text())
            assert gen['runtime']['counters'][2]==0
            videos.append(dict(run=name,job=gen['job'],generation_seconds=gen['seconds'],peak_allocated=gen['peak_allocated'],
                module_times=gen['runtime']['timings'],counters=gen['runtime']['counters'],decode_seconds=post['VAE_decode_wall_seconds'],encode_seconds=post['video_encode_seconds'],
                quality={window:{k:values[k] for k in ('lpips','ssim','psnr','window')} for window,values in post['quality'].items()},video=str((path.parent/'video.mp4').relative_to(ROOT))))
    assert len(videos)==9
    cache=json.loads((base/'cache_check/cache_check.json').read_text());resume=json.loads((base/'resume_check/resume_check.json').read_text())
    assert cache['passed'] and resume['passed']
    atomic(dest/'summary.json',dict(component_benchmark=bench,video_benchmark=video,videos=videos,cache_check=cache,resume_check=resume,single_video_speed_policy=True))
    for source,target in [('validation/validation.json','validation.json'),('cache_check/cache_check.json','cache_check.json'),('resume_check/resume_check.json','resume_check.json'),('prefix_checks.json','prefix_checks.json')]:shutil.copyfile(base/source,dest/target)
    dependencies=['local_experiments/review_v1/'+p for p in ('ops.py','runtime.py','model.py','graph_summary.py','cached_ops.py')]
    dependencies+=['local_experiments/'+p for p in ('gap_exact_round_kernel.py','broad_common.py','rollout_suite.py','stream_eval_common.py','high_sparse_experiment.py','e29_local_reader/core.py','e27_adaptive_query/core.py','e30_beta_diagnostics/run.py')]
    dependencies += [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py')]
    atomic(dest/'source_manifest.json',{p:digest(ROOT/p) for p in dependencies})
    rows=[r for r in bench['rows'] if r['component'] in ('512_swaps_set_only','page_readout','attention_full_mean_rank8','KV_update_one_tensor')]
    module_lines=[]
    for row in rows:
        if row['component']=='512_swaps_set_only' and row.get('batch')!=72:continue
        module_lines.append(f"| {row['component']} | {row['baseline']['wall_ms_median']:.4f} ms | {row['optimized']['wall_ms_median']:.4f} ms | {row['speedup']:.2f}× |")
    city=sorted([v for v in videos if v['job']['case']=='city_walk'],key=lambda v:(v['job']['chunks'],v['job']['method']))
    job_lines=[f"| {v['job']['chunks']} | {v['job']['method']} | {v['generation_seconds']:.2f} s | {v['decode_seconds']:.2f} s | {v['peak_allocated']/2**30:.2f} GiB |" for v in city]
    long=[v for v in city if v['job']['chunks']==41]
    stages=[]
    for v in long:
        timing=v['module_times'];stages.append('| '+v['job']['method']+' | '+' | '.join(f"{timing.get(k,{}).get('milliseconds',0)/1000:.2f}" for k in ('summary_and_encoding','route_and_correction','attention','clean_commit'))+' |')
    lines=['# 第一轮运行时优化记录','',
        '已将冻结算法接入常驻视频运行时，完成单视频速度对照、模块隔离计时、9/41-chunk 功能试跑和缓存/数值回归。新实现位于 `local_experiments/video_runtime_v1/`，没有覆盖旧研究实现。',
        '', '## 测速口径（按用户最新指引）','',
        '**以后只用单个视频测试加速效果，优先做单模块用时隔离，不扩展 prompt/seed/视频数量做测速。** batch 轴表示同一视频内的 heads × query tiles，不表示多个视频。',
        '', '整视频加速对照只使用 city_walk、noise seed701、9 chunks。原版和新版共享同一已加载模型及同一替代权重，均先做 3-chunk 预热；计时包含完整生成及 clean commit，不包含加载、VAE、视频编码或 latent IO。新版计时仍可能捕获其余有效页数的图，因此不是完全预热后的最佳值。每个实现只有一次计时，不作统计显著性声明。',
        '',f"原版 **{video['rows'][0]['median_seconds']:.2f} s** → 新版 **{video['rows'][1]['median_seconds']:.2f} s**，单次实测 **{video['speedup']:.2f}×**。allocated 峰值分别为 {video['rows'][0]['repetitions'][0]['peak_allocated']/2**30:.2f} / {video['rows'][1]['repetitions'][0]['peak_allocated']/2**30:.2f} GiB。GPU 为 A100-SXM4-40GB；同机其他 GPU 有训练/功能验证负载，这份工程记录不是严格独占整机的性能验收。",
        '', '历史组件计时使用 Set-only 的同一 E49 forest_run 输入以及 city_walk 的其余模块输入，生成于最新单视频限制之前；保留为已有隔离记录，不当作多视频整段加速。以后的入口默认统一 city_walk，并可只选择一个模块，不再跑输入矩阵。','', '已有模块隔离记录（同输入、预热后 wall median）：','', '| 模块 | 原版 | 优化 | 比值 |','|---|---:|---:|---:|',*module_lines,'', '## 实施内容','',
        '1. 历史摘要、Reader 85 系数、Adaptive Phi 32 系数及 B16 页面投影在 clean commit 时编码，随相同环形槽位更新；当前页每 step 编码一次，全部 query tiles 复用。响应与 beta 仍随 Query 计算。',
        '2. 原始 KV 保持原生 `[1,L,H,128]` 常驻；每步仅覆盖历史后的当前区域。精算直接用最终 LUT 读取原始 KV，没有整段历史 cat/repack 或逐 tile gather。页面顺序与旧淘汰语义一致。',
        '3. 512 交换共用背景、16/32 小型排除矩阵和五组正质量混合分母；FP64 端点增量保留微小正分，极端主导页移出采用独立归一化组混合兜底。接受规则为最高 raw gain 严格大于零，否则 no-op；没有为归一化额外跑 Dense。',
        '4. 精算复用现有 Page64/Q64 内核；补偿是完整 128 维均值矩阵乘 + rank8 detail 汇总/重建。正确转换精算 base-2 LSE 与补偿 natural LSE，并保留每个代表 16 tokens 的质量。',
        '5. Q64 R4 logits 同时供 M16 Risk/Reader 和最终补偿使用；投影按页缓存。Risk 保持 FP32，Reader 归一化/投影与集合评分采用 FP64。路由、候选、动作、no-op 和 LUT 在 GPU 图内执行；页面输入每层 forward 暂存一次，原始 KV 不进入图暂存区。',
        '6. 每 GPU 一个常驻 worker，一次生成一个视频；SQLite 唯一领取与身份校验、clean latent 恢复、分阶段 VAE/编码/指标、每参考任务一次 Dense 生成/解码及像素缓存复用均已接通。',
        '', '## 模块隔离计时','',
        '以下是用户提出单视频约束之前已完成的历史模块测量：评分来自一个 E49 forest_run 输入，其余模块来自 city_walk；不把两者合并成多视频加速比。以后默认模块基准也统一到 city_walk。旧 Set-only 去掉了 Independent 诊断，避免夸大收益。评分 batch72 为 12 heads × 6 tiles；页面读出和 Attention 同样为 12 heads × 6 tiles。',
        '', '| 模块 | 原实现 | 优化实现 | 加速 |','|---|---:|---:|---:|',*module_lines,
        '', '页面读出优化值为缓存命中后的 Query readout，encode-once 成本另外保存在原始 JSON；Attention 两边都包含 Q64 摘要 QK；KV 行只比较一个 K/V 张量的全历史 cat 与当前区写入。不能把这些组件倍数相乘作为端到端加速比。',
        '', '## 单视频成本台账','',
        '下列 9/41-chunk 值来自已有功能/质量试跑，含图初始化、latent IO 和 profiling；不是新增的多样本测速。固定同一 city_walk/seed701，分别生成三方法，记录各阶段成本。',
        '', '| chunks | 方法 | 生成含 IO | VAE 解码 | allocated 峰值 |','|---|---|---:|---:|---:|',*job_lines,
        '', '**优化稀疏路径目前仍慢于 Dense。** Full_Pset 的主要成本已集中到 GPU 路由/FP64 修正阶段；不能把原型→优化的加速理解成相对 Dense 加速。',
        '', '41-chunk 大模块 CUDA-event elapsed（秒，边界含宿主发射间隙；不包含整个模型的所有其他计算）：','',
        '| 方法 | 摘要/编码 | 路由/修正 | Attention | clean commit |','|---|---:|---:|---:|---:|',*stages,
        '', '## 正确性与边界','',
        f"- 14 个 E49 冻结输入全部保留最高动作与接受符号，包括约 1e-24 的微小正分及旧灾难提议；另外通过移出比背景大 1000 log units 的主导页测试。",
        f"- 缓存 Reader 最大差异 {validation['cached_reader_max_abs']:.3g}；参考 Risk 集合重合率 {validation['risk_overlap']:.6f}；局部 Attention NMSE {validation['attention_nmse']:.3g}，最大绝对误差 {validation['attention_max_abs']:.7f}。真实 3-chunk 另通过 29 层 × 4 steps 的 116 处局部门限检查。",
        '- CUDA Graph 与普通张量版本的同输入 3-chunk clean latent 逐位一致；只重放前两块 clean latent 后重新生成第三块，也逐位一致。',
        '- 41-chunk 后对 layers1/15/29 的全部 338 个幸存槽位独立重建 raw-KV 摘要和 Reader/Phi/投影缓存，12,168 个 head-pages 全部逐位一致，含部分旧 chunk 的保留槽位。三方法前两块 Dense latent 也逐位一致，验证 worker 重置。',
        f"- **原版与优化版的完整闭环并不逐位相同**：上述单视频 latent NMSE 为 {video['latent_nmse']:.6f}，max abs {video['latent_max_abs']:.6f}。实现改变了 FP32 规约和精算算术顺序；本轮未完成这项闭环差异的归因或质量等价审计，不能从局部误差门限推导视频质量等价。",
        '- 当前按照用户授权广播一份 l15h0 C pooling 权重，保持每层/head 的原 grouping、bank、basis 和 theta；Reader/Phi/B16 使用冻结旧模型。所有身份文件均标记 placeholder。不能据此评价正式全覆盖 C 或 Full_Pset 的视频收益；全覆盖训练独立继续。',
        '- 9 个既有功能视频已全部完成生成、解码、指标和编码；其中 41-chunk 三组有前/中/后 LPIPS、SSIM、PSNR。占位权重出现明显背景/主体漂移和局部伪影，图像指标不支持对正式方法质量作结论。',
        '', '## 现有视频与复现','',
        '- 41-chunk 三列对照：`artifacts/video_runtime_v1/long_pilot/videos/city_walk_s701/chunks41/comparison.mp4`（左 Dense / 中 C+Risk / 右 Full_Pset）。',
        '- 短片三列对照：`artifacts/video_runtime_v1/pilot/videos/city_walk_s701/chunks9/comparison.mp4`。',
        '- 真实单视频原版/新版计时：`artifacts/video_runtime_v1/video_benchmark/result.json`。',
        '- 模块隔离原始计时：`artifacts/video_runtime_v1/benchmark/benchmark.json`。',
        '- 紧凑汇总、数值/缓存/恢复检查与源码 SHA 保存在本目录；视频、clean latent 和 Dense 像素缓存保留在 artifacts，未复制进审阅结果目录。',
        '- 完整命令与以后的单视频测速约束见 `local_experiments/video_runtime_v1/README.md`。',
        '', '后续先用正式权重复核质量与闭环差异，再决定是否继续压缩 Full_Pset 评分内存流量。若测加速，仅用一个固定视频或单模块隔离计时，不再扩展样本矩阵。','']
    (dest/'REPORT.zh.md').write_text('\n'.join(lines))
    print(dest/'REPORT.zh.md')
if __name__=='__main__':run()
