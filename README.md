# Streaming KV Sparse Routing / SlimAV

本项目在 Self-Forcing 1.3B 的流式视频生成中使用页级稀疏注意力。每个查询块选择一部分 KV 页，读取原始 K/V 做精算；其余页通过学习得到的摘要参与注意力。当前 VBench 队列的 **intersection** 使用**原始稀疏模块权重、首帧锚点重定位和 10% 非锚点条件预算**。复现这条路径应从队列 worker 进入：它先创建 intersection 运行时，再调用 `queue_policy.attach(rt, engine, 10)` 安装预算表和重定位逻辑。[生成入口](reproduction/vbench_current/worker.py) · [条件策略](reproduction/vbench_current/queue_policy.py) · [新机器运行说明](reproduction/vbench_current/README.zh.md)

## 五档方法实际执行什么

下表使用代码中的层、头和去噪步编号，均从 0 开始。Full_Pset 指路由选页后的换页纠偏。表中的纠偏范围只决定哪些位置允许换页；其余稀疏层头仍执行选页、精算与摘要补偿。

| 队列方法 | 窗口与精算预算 | 选页与注意力实现 |
|---|---|---|
| Dense | 原生 18 个 latent 帧滑窗，窗内全部精算 | 同一优化主干的 `dense_native`，用作速度对照 |
| LightForcing | 上游短视频配置为 21 帧窗，长视频配置为 12 帧窗 | 同一 Self-Forcing EMA，运行上游 BF16 稀疏管线；A100 启用 RoPE、scale-shift 优化，worker 检查底层稀疏核调用 |
| base | 按全窗页数取 15% 向下取整，包含首帧 24 页及查询自身页；满窗 64/432 页 | 按风险补足名额，无 Full_Pset；A100 部署预设启用 SM80 固定预算 CUDA 融合核 |
| intersection | 固定首帧 24 页并重定位；其余页用 10% **平均**预算，按窗口、层、步、头查表 | 去噪步 0，在层 3、4、5、8、14、17、27 的头 3、4、9 上做分区纠偏；精算与补偿走 Triton 融合路径 |
| step | 按全窗页数取约 10%，满窗 43/432 页，保留查询自身页 | 去噪步 0 对所有稀疏层、所有头开放 Full_Pset；其余去噪步沿用风险选页，走 Triton 融合路径 |

base 的 15% 和 intersection 的 10% 使用不同的分母。首帧时间重定位和条件预算表由 intersection 的队列入口安装。[部署预设与内核开关](local_experiments/video_runtime_v1/accepted_tiers.py) · [intersection 纠偏范围](local_experiments/video_runtime_v1/presets/intersection.json) · [step 纠偏范围](local_experiments/video_runtime_v1/presets/step.json) · [LightForcing 适配](reproduction/vbench_current/light.py)

## 一个 chunk 如何生成

1. **生成与缓存。** 每个 chunk 含 3 个 latent 帧，每帧 1,536 个 token。依次执行 4 个去噪步，再以 clean latent 做一次 Dense 前向并提交 KV。前两个 chunk 使用 Dense；从第三个 chunk 开始，层 1–29 的去噪注意力可走稀疏路径，层 0 保留 Dense。[生成循环](local_experiments/rollout_suite.py) · [运行时分支与提交](local_experiments/video_runtime_v1/runtime.py)
2. **分组与摘要。** 每页 64 个 token，当前 chunk 共 72 页，满窗共 432 页。历史 KV 在 clean 提交时分组，并保存每页 4 个代表 K、V 均值、低秩细节和路由统计；当前 chunk 的摘要随去噪输入重算。[历史页分组](local_experiments/video_runtime_v1/grouping_dataflow.py) · [摘要计算](local_experiments/video_runtime_v1/summary_moments.py)
3. **选页与纠偏。** 每头、每 64 个查询 token 构成一个查询块。路由使用其中 16 个查询位置估计各页风险，生成精算页表。在启用 Full_Pset 的位置，LocalReader 给候选换页打分；intersection 只允许同分区内的合法交换，保护首帧和查询自身页，保持当前与历史的配额。[条件路由图](reproduction/vbench_current/graph.py) · [候选池与填充修正](reproduction/vbench_current/quota_pool.py) · [Reader 打分](local_experiments/video_runtime_v1/reader_reduce.py)
4. **精算与补偿。** 精算页读取原始 token 计算 QK 和双项 BF16 AV。其余页计算查询与代表 K 的注意力，用 V 均值和 rank-8 细节近似输出，再按两部分的 softmax 质量合并。[注意力分派](local_experiments/video_runtime_v1/legacy_components.py) · [Triton 融合核](local_experiments/review_v1/tuned_attention.py) · [base 的 CUDA 融合核](local_experiments/video_runtime_v1/csrc/fixed15_attention.cu)

## Intersection 的预算和首帧位置

设当前窗口有 W 页。首个 latent 帧的 24 页固定精算，当前 chunk 有 72 页，其余历史有 H = W − 24 − 72 页。若表项给当前 A 页、给历史 B 页，则当前精算比例 p = A/72，历史精算比例 q = B/H，实际精算页数为 **24 + A + B**。查询自身页计入 A。离线标定在平均预算约束下选择离散表项；同一层、同一步的各头共享总页数，A 和 B 可以因头而异。达到满窗后复用同一张表。[配额计算](reproduction/vbench_current/quota_counts.py) · [部署预算表](reproduction/vbench_current/policies.npz) · [标定与跨 prompt、seed 验证](reproduction/vbench_current/history/conditional_budget_v1/RESULTS.zh.md)

例如，读取 `10_window_step_layer_head` 的满窗、层 3、去噪步 0 表项，头 0 的当前/历史配额为 18/2 页，头 3 为 8/12 页。两者都精算 44 页，其中 24 页属于首帧。这个例子说明为什么不能把 10% 写成每次调用都固定精算 64 页。

从第 7 个 chunk 首次滚窗起，运行时每个 chunk 都从保存的原始首帧 K 重新施加时间 RoPE，使当前 3 帧到锚点的相对距离保持为 15、16、17。同时刷新首帧页摘要及对应层的纠偏编码。首帧固定精算从首次稀疏时起生效，时间重定位从首次滚窗起生效。[锚点保护](reproduction/vbench_current/anchor_policy.py) · [时间旋转和摘要刷新](reproduction/vbench_current/phase.py)

## 运行与核验

按复现说明准备模型、环境和独立工作目录后，从仓库根目录运行：

~~~bash
python -u reproduction/vbench_current/launch.py
python reproduction/vbench_current/status.py
~~~

队列共 5,760 条视频。每档含 768 条短视频（80 帧）和 384 条长视频（480 帧），均为 768×512、16 fps。生成与 VAE 解码分别计时；评分单独执行。东京 `long_0/701` 和猛犸 `long_1/702` 会核验完整 BF16 latent 的 SHA-256，用于检查实现一致性。长视频后段的画质结合成片中的接缝、颜色和背景连续性以及 VBench 分项结果判断。[复现命令与依赖](reproduction/vbench_current/README.zh.md) · [权重加载](local_experiments/video_runtime_v1/weights.py) · [权重与输入指纹](reproduction/vbench_current/files.json) · [评分入口](reproduction/vbench_current/score_batch.py)

## 项目训练权重

当前 10% 条件预算 intersection 加载原始训练权重。摘要和路由参数来自 [C 部署权重](artifacts/full_coverage_v1/training/C_s17/deployment.pt)，换页打分使用 [Full_Pset LocalReader](artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt)、[查询映射 AdaptivePhi](artifacts/e27_adaptive_query_v1/s17/selected.pt) 和 [B 投影](artifacts/e24_joint_representatives_v1/fit.pt)。加载器还会读取两份初始化文件。六份文件随 Git 克隆取得，复现入口会校验 SHA-256。[加载代码](local_experiments/video_runtime_v1/weights.py) · [部署文件清单](reproduction/vbench_current/files.json)

以下是另行保存的消融权重，可用于同预算的训练前后对照。每组目录包含 summary.pt 和 reader.pt；补偿方向缩放实验单独保存完整摘要权重。

| 训练口径 | 10% 非锚点预算 | 15% 非锚点预算 |
|---|---|---|
| 固定配额下两阶段重训练 | [摘要与 Reader](reproduction/vbench_current/history/rebased_retrain_v1/budget_11) | [摘要与 Reader](reproduction/vbench_current/history/rebased_retrain_v1/budget_21) |
| 条件预算表下两阶段微调 | [摘要与 Reader](reproduction/vbench_current/history/conditional_budget_v1/budget_10) | [摘要与 Reader](reproduction/vbench_current/history/conditional_budget_v1/budget_15) |

原 B 策略的方向缩放候选及九份消融文件的指纹也已归档。[方向缩放候选](reproduction/vbench_current/history/intersection_b_tune01/candidate.pt) · [消融权重 SHA-256](reproduction/vbench_current/experimental-weights.json)

## 历史版本与其他实验

- 2026-11-02 三场景比较使用当时的 GapGainVector64Runtime。[对照报告](docs/streaming_evaluation_20261102/report.zh.md) · [源码说明](docs/SOURCE_RELEASE_20261102.md)
- 早期 v0.8 的三层稀疏、Router/LoRA 训练，以及 Opt30–Opt32 的实验有各自的协议和权重。[迁移与继续训练](REBUILD.md) · [Opt 实验总结](docs/EXPERIMENT_SUMMARY.txt) · [11 月 13 日权重发布](reproduction/weights_20261113/README.md)
- [Self-Forcing 原项目说明](README_SELF_FORCING.md) · [许可证](LICENSE)
