# 第一轮运行时优化记录

已将冻结算法接入常驻视频运行时，完成单视频速度对照、模块隔离计时、9/41-chunk 功能试跑和缓存/数值回归。新实现位于 `local_experiments/video_runtime_v1/`，没有覆盖旧研究实现。

## 测速口径（按用户最新指引）

**以后只用单个视频测试加速效果，优先做单模块用时隔离，不扩展 prompt/seed/视频数量做测速。** batch 轴表示同一视频内的 heads × query tiles，不表示多个视频。

整视频加速对照只使用 city_walk、noise seed701、9 chunks。原版和新版共享同一已加载模型及同一替代权重，均先做 3-chunk 预热；计时包含完整生成及 clean commit，不包含加载、VAE、视频编码或 latent IO。新版计时仍可能捕获其余有效页数的图，因此不是完全预热后的最佳值。每个实现只有一次计时，不作统计显著性声明。

原版 **244.57 s** → 新版 **25.46 s**，单次实测 **9.61×**。allocated 峰值分别为 7.89 / 8.25 GiB。GPU 为 A100-SXM4-40GB；同机其他 GPU 有训练/功能验证负载，这份工程记录不是严格独占整机的性能验收。

历史组件计时使用 Set-only 的同一 E49 forest_run 输入以及 city_walk 的其余模块输入，生成于最新单视频限制之前；保留为已有隔离记录，不当作多视频整段加速。以后的入口默认统一 city_walk，并可只选择一个模块，不再跑输入矩阵。

已有模块隔离记录（同输入、预热后 wall median）：

| 模块 | 原版 | 优化 | 比值 |
|---|---:|---:|---:|
| 512_swaps_set_only | 421.2149 ms | 5.1508 ms | 81.78× |
| page_readout | 5.5487 ms | 0.5464 ms | 10.15× |
| attention_full_mean_rank8 | 10.9497 ms | 1.4896 ms | 7.35× |
| KV_update_one_tensor | 0.2593 ms | 0.0566 ms | 4.58× |

## 实施内容

1. 历史摘要、Reader 85 系数、Adaptive Phi 32 系数及 B16 页面投影在 clean commit 时编码，随相同环形槽位更新；当前页每 step 编码一次，全部 query tiles 复用。响应与 beta 仍随 Query 计算。
2. 原始 KV 保持原生 `[1,L,H,128]` 常驻；每步仅覆盖历史后的当前区域。精算直接用最终 LUT 读取原始 KV，没有整段历史 cat/repack 或逐 tile gather。页面顺序与旧淘汰语义一致。
3. 512 交换共用背景、16/32 小型排除矩阵和五组正质量混合分母；FP64 端点增量保留微小正分，极端主导页移出采用独立归一化组混合兜底。接受规则为最高 raw gain 严格大于零，否则 no-op；没有为归一化额外跑 Dense。
4. 精算复用现有 Page64/Q64 内核；补偿是完整 128 维均值矩阵乘 + rank8 detail 汇总/重建。正确转换精算 base-2 LSE 与补偿 natural LSE，并保留每个代表 16 tokens 的质量。
5. Q64 R4 logits 同时供 M16 Risk/Reader 和最终补偿使用；投影按页缓存。Risk 保持 FP32，Reader 归一化/投影与集合评分采用 FP64。路由、候选、动作、no-op 和 LUT 在 GPU 图内执行；页面输入每层 forward 暂存一次，原始 KV 不进入图暂存区。
6. 每 GPU 一个常驻 worker，一次生成一个视频；SQLite 唯一领取与身份校验、clean latent 恢复、分阶段 VAE/编码/指标、每参考任务一次 Dense 生成/解码及像素缓存复用均已接通。

## 模块隔离计时

以下是用户提出单视频约束之前已完成的历史模块测量：评分来自一个 E49 forest_run 输入，其余模块来自 city_walk；不把两者合并成多视频加速比。以后默认模块基准也统一到 city_walk。旧 Set-only 去掉了 Independent 诊断，避免夸大收益。评分 batch72 为 12 heads × 6 tiles；页面读出和 Attention 同样为 12 heads × 6 tiles。

| 模块 | 原实现 | 优化实现 | 加速 |
|---|---:|---:|---:|
| 512_swaps_set_only | 421.2149 ms | 5.1508 ms | 81.78× |
| page_readout | 5.5487 ms | 0.5464 ms | 10.15× |
| attention_full_mean_rank8 | 10.9497 ms | 1.4896 ms | 7.35× |
| KV_update_one_tensor | 0.2593 ms | 0.0566 ms | 4.58× |

页面读出优化值为缓存命中后的 Query readout，encode-once 成本另外保存在原始 JSON；Attention 两边都包含 Q64 摘要 QK；KV 行只比较一个 K/V 张量的全历史 cat 与当前区写入。不能把这些组件倍数相乘作为端到端加速比。

## 单视频成本台账

下列 9/41-chunk 值来自已有功能/质量试跑，含图初始化、latent IO 和 profiling；不是新增的多样本测速。固定同一 city_walk/seed701，分别生成三方法，记录各阶段成本。

| chunks | 方法 | 生成含 IO | VAE 解码 | allocated 峰值 |
|---|---|---:|---:|---:|
| 9 | c_pset | 84.01 s | 4.30 s | 8.89 GiB |
| 9 | c_risk | 24.99 s | 4.31 s | 8.24 GiB |
| 9 | dense | 11.15 s | 4.66 s | 8.04 GiB |
| 41 | c_pset | 469.79 s | 19.94 s | 8.89 GiB |
| 41 | c_risk | 145.54 s | 19.92 s | 8.24 GiB |
| 41 | dense | 50.92 s | 20.35 s | 8.04 GiB |

**优化稀疏路径目前仍慢于 Dense。** Full_Pset 的主要成本已集中到 GPU 路由/FP64 修正阶段；不能把原型→优化的加速理解成相对 Dense 加速。

41-chunk 大模块 CUDA-event elapsed（秒，边界含宿主发射间隙；不包含整个模型的所有其他计算）：

| 方法 | 摘要/编码 | 路由/修正 | Attention | clean commit |
|---|---:|---:|---:|---:|
| c_pset | 7.68 | 366.15 | 63.99 | 5.56 |
| c_risk | 6.62 | 42.26 | 64.17 | 5.87 |
| dense | 0.00 | 0.00 | 19.78 | 5.86 |

## 正确性与边界

- 14 个 E49 冻结输入全部保留最高动作与接受符号，包括约 1e-24 的微小正分及旧灾难提议；另外通过移出比背景大 1000 log units 的主导页测试。
- 缓存 Reader 最大差异 9.54e-07；参考 Risk 集合重合率 1.000000；局部 Attention NMSE 2.92e-09，最大绝对误差 0.0039062。真实 3-chunk 另通过 29 层 × 4 steps 的 116 处局部门限检查。
- CUDA Graph 与普通张量版本的同输入 3-chunk clean latent 逐位一致；只重放前两块 clean latent 后重新生成第三块，也逐位一致。
- 41-chunk 后对 layers1/15/29 的全部 338 个幸存槽位独立重建 raw-KV 摘要和 Reader/Phi/投影缓存，12,168 个 head-pages 全部逐位一致，含部分旧 chunk 的保留槽位。三方法前两块 Dense latent 也逐位一致，验证 worker 重置。
- **原版与优化版的完整闭环并不逐位相同**：上述单视频 latent NMSE 为 0.180343，max abs 4.046875。实现改变了 FP32 规约和精算算术顺序；本轮未完成这项闭环差异的归因或质量等价审计，不能从局部误差门限推导视频质量等价。
- 当前按照用户授权广播一份 l15h0 C pooling 权重，保持每层/head 的原 grouping、bank、basis 和 theta；Reader/Phi/B16 使用冻结旧模型。所有身份文件均标记 placeholder。不能据此评价正式全覆盖 C 或 Full_Pset 的视频收益；全覆盖训练独立继续。
- 9 个既有功能视频已全部完成生成、解码、指标和编码；其中 41-chunk 三组有前/中/后 LPIPS、SSIM、PSNR。占位权重出现明显背景/主体漂移和局部伪影，图像指标不支持对正式方法质量作结论。

## 现有视频与复现

- 41-chunk 三列对照：`artifacts/video_runtime_v1/long_pilot/videos/city_walk_s701/chunks41/comparison.mp4`（左 Dense / 中 C+Risk / 右 Full_Pset）。
- 短片三列对照：`artifacts/video_runtime_v1/pilot/videos/city_walk_s701/chunks9/comparison.mp4`。
- 真实单视频原版/新版计时：`artifacts/video_runtime_v1/video_benchmark/result.json`。
- 模块隔离原始计时：`artifacts/video_runtime_v1/benchmark/benchmark.json`。
- 紧凑汇总、数值/缓存/恢复检查与源码 SHA 保存在本目录；视频、clean latent 和 Dense 像素缓存保留在 artifacts，未复制进审阅结果目录。
- 完整命令与以后的单视频测速约束见 `local_experiments/video_runtime_v1/README.md`。

后续先用正式权重复核质量与闭环差异，再决定是否继续压缩 Full_Pset 评分内存流量。若测加速，仅用一个固定视频或单模块隔离计时，不再扩展样本矩阵。
