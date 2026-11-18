# 仓库现状与工作交接（2026-11-18）

[当前 105 帧 base 各模块 profile、理论耗时与差距](evidence/video_runtime_v1/PROFILE_MODULES_105.zh.md) · [模块表 CSV](evidence/video_runtime_v1/PROFILE_MODULES_105.csv)


**后续等值优化**：Risk 八维投影与消费合并、分组配对 gather 与初始 one-hot 直接取值已成为本地默认。60 条跨场景稀疏轨迹的 latent 均逐位一致。[验收及目标差距](../../artifacts/video_runtime_v1/risk_group_bundle_v1/PROGRESS.zh.md)。

**精算 AV 决定**：20 组跨场景验证后，用户拒绝单项 AV。base、intersection、step 继续使用双项 AV。见[验证报告](../../artifacts/video_runtime_v1/exact_head_major_v1/validation/REPORT.zh.md)。

**11 月 18 日最新更新**：路由 QK 已启用 BF16 直读和 FP32 缩放融合。用户选择更快实现；base 的轨迹差异、另外两档的单视频一致性、当前速度及回退开关见[QK 验收报告](../../artifacts/video_runtime_v1/scaled_route_qk_v1/REPORT.zh.md)。

**此前 RMSNorm 更新**：已恢复一次针对性优化，单 kernel RMSNorm 通过模块和三个稀疏档位、原生 Dense 的短视频验收，部署默认开启 `rms_row`。下文暂停点与速度表为此前快照。最新速度、误差及复现方式见[本轮报告](../../artifacts/video_runtime_v1/rms_row_v1/REPORT.zh.md)。

## 1. 先掌握当前结论

项目在流式视频生成中，用学习到的历史页摘要和Risk初选分配稀疏Attention精算预算，再可选地用Full_Pset做一次集合感知交换。当前已经有可完整生成视频的GPU运行时、全覆盖部署权重、常驻批量worker和分阶段质量评价。

工作已从早期局部研究，推进到**完整视频质量验证、推理预算选择和运行时优化**。最近按用户要求完成了额外两轮“profile→理论工作量分析→实现→测试验收”，优化目标现已暂停。本次只整理交接，不启动训练、第三轮优化或视频任务。

当前保留三个档位：**关闭Full_Pset、原交集、按step选择**。最新接受的改动是摘要池化与统计共享中间结果；在线Risk概率原型没有稳定视频收益且改变部分轨迹，保持关闭。

这里“已部署”指已成为本地运行时默认配置，并非已经发布到远端。当前Git HEAD为`2b657a7`；后续优化代码和报告有大量尚未提交的改动。

## 2. 推荐阅读顺序与信息优先级

| 顺序 | 入口 | 用途 |
|---|---|---|
| 1 | 本文 | 理解当前算法、三档配置、工作终点和重启边界 |
| 2 | [当前默认配置](../../local_experiments/video_runtime_v1/deployment.py)、[三档入口](../../local_experiments/video_runtime_v1/accepted_tiers.py) | 确认实际运行的权重、开关和策略 |
| 3 | [最近两轮暂停交接](../../artifacts/video_runtime_v1/post_rope_round2/PAUSE_HANDOVER.zh.md) | 最新速度、验收及未接受原型 |
| 4 | [运行时](../../local_experiments/video_runtime_v1/runtime.py)、[路由图](../../local_experiments/video_runtime_v1/pipeline_graph.py) | 沿数据生命周期理解实现 |
| 5 | [5 prompt × 4 seed验证](../../artifacts/video_runtime_v1/intersection_validation_v1/REPORT.zh.md) | 理解为何保留原交集和step档 |
| 6 | [历史交接](PROJECT_HANDOVER_CURRENT.zh.md)、[历史研究阅读顺序](READING_ORDER.zh.md) | 按需追溯研究动机、失败分支和旧实验 |

实际配置、对应实验的result/identity及报告一起判断。不要仅凭历史README中的“当前”“下一步”判断今天的状态。`READING_ORDER.zh.md`的主体对应截至E42的研究阶段；其中“尚无可靠在线修正器”等文字应放回当时的实验范围，不能覆盖后续全覆盖训练和视频证据。历史报告的性能也不能直接当作新基线性能。

原用户引用的`PROJECT_HANDOVER_CURRENT.md`，当前仓库对应文件为`PROJECT_HANDOVER_CURRENT.zh.md`。

## 3. 当前方法究竟执行什么

固定算法配置：Page64、Q64、R4、完整128维V均值＋rank8细节、M16 probes、B16投影、10%精算页预算；执行Full_Pset的事件中，评估16个移出候选×32个加入候选，共512次交换，并保留no-op，最高代理gain大于0才接受。

数据流如下：

1. 历史KV常驻，当前KV写入预留区域；避免每个去噪step拼接整段历史。
2. 当前noisy页每步更新摘要；历史clean页在commit时构造摘要，并在需要Full_Pset的配置中维护Reader页面编码。
3. M16 Query计算摘要QK和Risk，按预算得到精算页初选。
4. 如果当前层/head/step在Full_Pset范围内，使用缓存的页面系数、Reader及共享背景评分评估512个交换；否则直接使用初选。
5. 精算kernel通过页LUT直接读原始KV；补偿保留完整V均值和rank8细节，排除精算页，两分支按LSE合并。
6. clean forward之后分组和提交，保持历史页生命周期。clean commit不是可任意跳过的额外开销。

Full_Pset是集合修正模块，用户消息中的Full_qset对应本模块。M16是Query采样数，B16是输出投影维度，不能与Page64混淆。

## 4. 权重与三个保留档位

当前部署由`deployment.py`显式指定：

- C摘要模型：[C_s17/deployment.pt](../../artifacts/full_coverage_v1/training/C_s17/deployment.pt)。已经走全覆盖部署，不再使用最初E24的12个layer–head覆盖或单权重广播作为正式基线。
- Reader：[Full_Pset_s17/selected.pt](../../artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt)。
- Adaptive Phi：`artifacts/e27_adaptive_query_v1/s17/selected.pt`。
- 配套B16：`artifacts/e24_joint_representatives_v1/fit.pt`。
- 精确依赖及SHA见[weights.py](../../local_experiments/video_runtime_v1/weights.py)和每次运行的identity。不要只复制两个主checkpoint就假定依赖齐全。

`weights.load()`无checkpoint时仍有E24广播回退；这不是当前部署入口的正常调用方式。基础档可以不加载Reader/Phi/B。全覆盖训练代码在`local_experiments/full_coverage/`，训练产物在`artifacts/full_coverage_v1/training/`。本交接没有重审其他并行训练任务的实时状态。

| 档位 | Full_Pset执行范围 | 配置入口 |
|---|---|---|
| `base` | 完全关闭，C＋Risk | `accepted_tiers.config('base')` |
| `intersection` | layers={3,4,5,8,14,17,27}且heads={3,4,9}且noisy step=0 | [intersection.json](../../local_experiments/video_runtime_v1/presets/intersection.json) |
| `step` | 所有适用层/head，仅noisy step=0 | [step.json](../../local_experiments/video_runtime_v1/presets/step.json) |

layer/head/step均使用代码中的索引，step0是第一个noisy step，不是最后一步。层0维持原生路径；“所有适用层”不能读成对原生层0也强行运行集合修正。交集是三个条件同时满足，不是并集。

`deployment.CURRENT`保留研究用的full-on默认。**复现用户保留三档应使用accepted_tiers，而不是裸用current()就假定是原交集。** 批量入口通过`--pset-policy`传策略文件，任务method决定是否启用修正；不同策略应使用独立且明确的任务身份/输出目录。

## 5. 研究与质量工作已到哪里

已完成单场景两seed的逐层、逐head、逐step独立开启试验，以及按收益筛选预算；随后在原交集预算附近进行了搜索，最终两版搜索结果与原交集及单轴策略共同参加冻结验证。

- [单轴消融](../../artifacts/video_runtime_v1/pset_scope_v1/REPORT.zh.md)。独立性是实验选择假设，不是已证明层/head/step之间没有相互作用。
- [交集预算搜索](../../artifacts/video_runtime_v1/intersection_search_v1/REPORT.zh.md)：city_walk、seed701/702，341个不同配置、682个视频。
- [冻结验证](../../artifacts/video_runtime_v1/intersection_validation_v1/REPORT.zh.md)：5 prompts × 4 seeds，每组7种方法，共140视频。

验证prompt为kitchen_cook、forest_run、portrait_turn、reef_fish、tennis_match，seed703–706。它们没有参与本次策略搜索；不能据此说没有参与模型训练。

冻结验证相对Dense的平均LPIPS：原交集0.36448、搜索均值最优0.36673、搜索双seed均衡0.36727、按step0.36377。两版搜索没有在这批验证中超过原交集，用户最终保留原交集、step0以及基础档。不要把搜索集最优配置直接替换成当前部署默认。

这批历史质量结果与后续单场景实现验收是不同实验。算法相对Dense的误差、优化实现相对旧实现的误差，也必须分开报告。当前优化工作主要负责后者。

## 6. 代码地图：从入口沿执行路径阅读

以下运行时文件均位于`local_experiments/video_runtime_v1/`。

| 文件/目录 | 当前职责 |
|---|---|
| `deployment.py`、`weights.py` | 共用配置、权重来源、初始化与身份记录 |
| `accepted_tiers.py`、`presets/` | 用户保留三档及策略 |
| `runtime.py`、`../review_v1/runtime.py::PageStore` | forward/clean commit、历史与当前页缓存 |
| `pipeline_graph.py` | M16路由、Risk、可选修正、最终LUT及CUDA Graph缓存 |
| `page.py`、`reader_reduce.py`、`reader_split.py` | 静态页面编码、Reader标量/向量阶段及动作状态 |
| `scores.py`、`endpoint_den.py`、`exclude_scan.py` | 候选、共享背景、稳定正质量分母和排除扫描 |
| `risk_projected.py`、`logmass_cached.py`、`tensor_bmm.py` | 当前Risk、log质量复用及QK |
| `legacy_components.py` | 接入旧的快速精算和完整均值补偿；不是把整个旧算法替换进来 |
| `summary_moments.py` | 最新接受的池化/代表统计共享 |
| `grouping_dataflow.py`、`group_softmax.py` | clean分组及保持归约顺序的softmax |
| `backbone_ops.py`、`block_pointwise.py`、`rms_dataflow.py`、`rms_mean.py`、`native_rope.py` | Dense/稀疏共用主干优化 |
| `batch.py`、`worker.py`、`queue.py`、`manifest.py` | 每GPU常驻worker、队列、恢复及生成/解码/指标分阶段 |
| `benchmark_hotspots.py`、`profile_retained.py` | 短视频计时、完整chunk profile |
| `benchmark_fair_dense.py` | 原生Dense与保留三档的公平主干测速 |
| `check_*.py`、`report_*.py`、`quality_*.py` | 对应实验的数值契约、报告及视频误差验证 |

精算底层在`local_experiments/gap_exact_round_kernel.py`，补偿及合并在`local_experiments/review_v1/tuned_attention.py`。摘要原路径在`local_experiments/review_v1/split_summary.py`。不能只读video_runtime_v1就认定所有kernel都定义在这里。

其他目录：`local_experiments/e*`是研究演进；`streaming_sparse/v08`提供主干/协议等依赖，也有独立训练状态；`artifacts`是原始结果和大文件；`docs/research/evidence`是选取的研究证据快照，不保证含所有最新运行产物。

## 7. 已接受优化与关闭分支

当前关键开启项：`summary_moments`、`native_rope`、`qk_tile64`、`rms_mean`、`rms_dataflow`、`block_pointwise`、`grouping_spatial`、`grouping_dataflow`、`route_m16`、`projected_risk`、`cached_logmass`、Reader split/A-only、FP32输出缓存、BF16页面系数存储、端点分母复用、FP32扫描消费。

这里FP32输出存储和BF16系数存储针对原来已有数值的容器冗余；不要因此推断任意新低精度改动都无损。精算和补偿保持此前用户接受的快速实现；此前接受过的近似也是基线的一部分。

| 分支 | 状态与原因 | 证据目录（artifacts/video_runtime_v1/下） |
|---|---|---|
| 摘要统计复用 | 默认开启，模块及三档视频逐位一致 | `post_rope_round1/` |
| Risk在线概率 | 关闭；模块收益未转成视频收益，部分轨迹变化 | `post_rope_round2/` |
| 精算acc复用 `exact_fold` | 关闭；模块1.09–1.12×，微小舍入差在视频中明显放大 | `exact_pipeline_v1/` |
| 全block CUDA Graph | 默认关闭；少量热运行收益，冷捕获成本明显 | `block_retained_v1/` |
| Reader小WORK分批P2 | 已结束；512/1024行没有稳定完整路由收益 | `p2_bounded_v1/` |
| PAGE32、lazy、vector G2 | 当前关闭，不当作尚未尝试的大收益方向 | 对应历史报告及CURRENT |
| tail生命周期/布局原型 | 未部署，不重启相同参数筛查 | `tail_liveness_v1/`、`tail_layout_v1/` |

## 8. 最新速度与最后两轮结果

固定city_walk/seed701，9chunks、105帧、768×512、16FPS、history360、tile72，4次noisy step。每进程完整预热后5次生成取中位数，排除VAE、编码、文件IO和冷编译。

| 档位 | 当前生成/s | 同卡优化Dense/s | 相对Dense | 最后一项已接受优化加速 |
|---|---:|---:|---:|---:|
| base（GPU6） | 6.407052 | 6.894364 | 1.07606× | 1.00765× |
| intersection（GPU7） | 6.450988 | 6.835182 | 1.05956× | 1.00952× |
| step（GPU5） | 7.111215 | 6.843190 | 0.96231× | 1.00953× |

这只是固定短视频计时，不是20个样本的最新平均质量/速度结果，也不是含VAE总耗时。Dense已共享逐元素、RMS和RoPE优化；此前未给Dense启用这些公共优化的比值不能沿用。批量worker的dense任务仍经过VideoRuntime的clean提交流程，和这里`dense_native`原生测速入口不同。

第一轮：池化与代表统计原本重复读取KV并算均值。共享统计后，每次摘要净减少约20.984MB逻辑读写，模块1.17–1.20×；三档短视频latent/计数逐位一致，`summary_moments=True`已接受。

第二轮：把Risk整张概率矩阵改为行max/den及消费时生成概率。模块约1.17–1.37×，完整路由仅1.003–1.010×。profile发现新增独立QK缩放kernel约15.860ms/chunk，基本吃掉归一化部分的节省；视频没有稳定提速。

第二轮相对第一轮同档输出：base latent NMSE0.33313、LPIPS0.220271；intersection NMSE0.004529、LPIPS0.004387；step逐位一致。属于FP32归约顺序变化及闭环放大，没有新增量化。base从chunk2、intersection从chunk7开始不同。完整PSNR/SSIM/RGB及并排视频见[第二轮报告](../../artifacts/video_runtime_v1/post_rope_round2/MODULE_DISTANCE.zh.md)。该原型保持关闭，没有把“可以训练补回”当作验收依据。

## 9. 复现入口与资源

仓库：`/home/shuchenweng/jzk/streaming_sparse`；工作规则：父目录`AGENTS.md`。

Python：`/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python`。快速模型资产：`/tmp/streaming_sparse-0/runtime`。现有环境使用8张A100 40GB、Torch2.6.0+cu124、Triton3.2.0；之前使用NVCC12.2编译小型CUDA扩展。重启后先检查这些本机路径，克隆仓库本身不包含全部模型、数据和视频。

下面命令用于后续明确恢复工作时执行，本次没有执行：

```bash
cd /home/shuchenweng/jzk/streaming_sparse
CUDA_VISIBLE_DEVICES=6 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python \
  -m local_experiments.video_runtime_v1.accepted_tiers \
  --tier base --reps 5 --skip-profile \
  --local-assets /tmp/streaming_sparse-0/runtime \
  --output artifacts/video_runtime_v1/FRESH_RUN/base
```

将tier替换为intersection/step即可测其他保留档。输出目录应取新的名称，以免混入不同配置的结果。

```bash
CUDA_VISIBLE_DEVICES=6 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python \
  -m local_experiments.video_runtime_v1.profile_retained \
  --tier base --output artifacts/video_runtime_v1/FRESH_PROFILE/base
```

当前例行验收只做短视频；15s仅在用户明确要求质量测试时跑，30s不跑。先隔离模块、验证输入输出和完整路由，再做一轮短视频验收。GPU计时必须预热；不能把冷编译、IO或不同GPU的绝对数值混作严格A/B。大文件及编译缓存优先使用快速/tmp资产区。

## 10. 暂停点、Git状态与下次工作

**目标保持暂停。** 本次文档整理不代表恢复优化，也不追加第三轮。

截至本次核查，HEAD仍为`2b657a7`。该提交包含三档选择、搜索和冻结验证。上次推送因VSCode Git认证socket失效而未完成；本次没有重新尝试推送，不能说远端已更新。后续摘要、主干、QK、RoPE等优化有未提交代码和报告，工作区同时含其他研究文档、v08训练checkpoint/status修改及未跟踪文件。不要`git add .`、清理整个工作区或覆盖这些训练产物。

恢复工作时建议顺序：

1. 先确认用户希望继续优化、提交整理，还是扩大质量验证；三档选择已经定下，不重复做此前搜索。
2. 如要提交，按运行时代码、对应小型证据和文档明确选文件，核对大文件与其他训练修改；修复认证后再推送。
3. 如继续Risk优化，先处理QK缩放和归一化被自定义算子边界拆开的数据流，并核对编译后的实际融合和原softmax归约树；不要只按概率矩阵字节数预测端到端收益。
4. 重新profile当前已接受基线。最新满窗口chunk仍由GEMM、FlashAttention、稀疏精算、补偿和路由主导；之前已试过的精算/tail参数方案不再当作新思路。
5. 任何新实现只与当前同档基线比较。局部输出或单份capture的mask一致不等于整个视频轨迹一致；有疑问则记录并保持关闭，不需要用户当场做验收决定。

最后两轮完整检查、代码SHA和证据清单见[completion_audit.json](../../artifacts/video_runtime_v1/post_rope_round2/completion_audit.json)。

## 11. 恢复优化后的运行时验收

在后续用户指令下恢复优化。已启用 clean 四流分组和尾部融合，保留双项 AV。三档单视频及多视频配对、60 条冻结 latent、显存及同卡 Dense 对照见[验收记录](../../artifacts/video_runtime_v1/tail_merge_overlap_v1/PROGRESS.zh.md)。GPU7 的 105 帧 base 为 6.015 秒，目标约 4.85 秒。

随后启用同分排序修正与单 tile 输出复用。相对四流分组版本，60 对整视频计时全部更快，60 个冻结样本 latent 相同。见[路由与输出复用验收](../../artifacts/video_runtime_v1/tie_direct_v1/PROGRESS.zh.md)。

FFN 保留 BF16 舍入并融合 GELU，小分组幂迭代匹配原 cuBLAS 的 FP32 次序，连续槽位直接写入。三项均已启用，两个视频计时组的 60 对全部更快，60 条冻结轨迹逐字节相等。同卡同主干 Dense 6.700 秒，base 5.814 秒。见[模块与视频验收](../../artifacts/video_runtime_v1/ffn_group_commit_v1/PROGRESS.zh.md)。

随后启用 LayerNorm 与帧调制融合，以及 32/64/128 token 分组的中心化融合。GPU7 同主干 Dense 6.614 秒、base 5.723 秒。见[验收记录](../../artifacts/video_runtime_v1/norm_raw_v1/PROGRESS.zh.md)。后续按用户最新要求，先做模块和单视频配对，收益小时重复同一视频，只有波动较大时才做多视频验证。

大分组幂迭代融合已启用。默认推理入口各测 5 次，GPU7 base 中位 5.64029 秒。沿用同卡同主干 Dense 中位 6.61363 秒，加速 1.17257×；目标 1.40034× 对应 4.72287 秒，还需节省 0.91742 秒。三档最终 latent 均与上次部署逐字节一致。[单视频验收记录](../../artifacts/video_runtime_v1/root_group_direction_v1/PROGRESS.zh.md)

已启用分组特征融合与共享内存 softmax。单视频 24 对全部更快，latent 逐字节一致。GPU7 默认入口 base 5.62328 秒，较同主干 Dense 加速 1.17612×。[分组特征验收](../../artifacts/video_runtime_v1/group_feature_bundle_v1/PROGRESS.zh.md)

路由 probe 直接收集到图输入，并省去投影 Risk 未读取的输入拷贝。单视频 18 对均更快，三档 latent 均相同。默认入口 base 5.58423 秒，同卡加速 1.18434×。[路由输入验收](../../artifacts/video_runtime_v1/projected_route_inputs_v1/PROGRESS.zh.md)

Q/K RMSNorm 和 FP64 RoPE 融合在三档及原生 Dense 同时启用，单视频 24 对均更快且 latent 字节相同。重测同卡 Dense 6.58614 秒、base 5.53192 秒，当前 1.19057×。[RMSNorm 与 RoPE 验收](../../artifacts/video_runtime_v1/rms_rope_v1/PROGRESS.zh.md)

固定预算页选择和按分支匹配的 Risk 行归约已启用。单视频 18 对速度与 latent 均通过。GPU7 同主干原生 Dense 6.58614 秒，base 5.48056 秒，加速 1.20173×。1.40034× 对应 4.70324 秒，尚差 0.77731 秒。[路由验收与剩余热点](../../artifacts/video_runtime_v1/page_route_v1/PROGRESS.zh.md)

CUDA 代表池化已启用。105 帧单视频三档 18 对全部更快，配对节省 82–112 ms，最终 latent 逐字节相同。默认入口 GPU7 base 5.39238 秒，沿用同卡原生 Dense 6.58614 秒，当前 1.22138×，距 4.70324 秒目标还差 0.68914 秒。GPU 时间线中代表池化从 169 ms 降至 61 ms。[模块、单视频和精算原型记录](../../artifacts/video_runtime_v1/cuda_pool_v1/PROGRESS.zh.md)

已启用稳定分组排序与 gather 融合、大组 CUB 基数排序、补偿直接读取摘要缓存以及匹配路由分支的 Risk 上界核。合并版 105 帧单视频三档共 24 对验证，latent 均逐字节相同。GPU7 base 5.35537 秒，相对同卡原生 Dense 1.22982×，距目标还差 0.65213 秒。[验收记录](../../artifacts/video_runtime_v1/group_radix_tail_bound_v1/PROGRESS.zh.md)

已启用按 128 页分块的 FP32 QK/Risk，并在候选页分数相对间隔 ≤4e-5 时重算该查询。修正前 intersection latent NMSE 0.09656，修正后三档单视频共 30 对全部更快、latent 字节一致。GPU7 默认 base 5.32914 秒，相对同卡原生 Dense 为 1.23587×，距离目标还差 0.62590 秒。[验收与适用范围](../../artifacts/video_runtime_v1/rechecked_mass_v1/PROGRESS.zh.md)

已启用页精算、选页掩码和补偿的单 kernel 路径。四种窗口的组合模块均提速，三档单视频 30 对全部更快，最终 latent 逐字节相同。GPU7 默认 base 5.31956 秒，相对同卡原生 Dense 1.23810×，距目标还差 0.61632 秒。[验收与硬件计数](../../artifacts/video_runtime_v1/fused_exact_tail_v1/PROGRESS.zh.md)

默认启用 KV 投影和稀疏当前页摘要直接写缓存。GPU7 原生 Dense 更新为 6.56527 秒。两项优化均启用后，默认入口 5 次中位数 base 5.25095 秒、intersection 5.32040 秒、step 5.95632 秒，三档 latent 与上次部署逐字节相同。base 相对同卡 Dense 提速 1.25030×，距 4.68834 秒目标还差 0.56261 秒。[配对验收、硬件计数与正式入口](../../artifacts/video_runtime_v1/summary_window_v1/PROGRESS.zh.md)

启用两处残差与归一化融合、满窗口未纠正路由的分层常驻图。默认入口各预热后 5 次的中位数为 base 5.20158 秒、intersection 5.23170 秒、step 5.94019 秒，三档最终 latent 均与上一部署逐字节一致。GPU7 原生 Dense 同样启用残差融合，为 6.54073 秒；base 加速 1.25745×，距 4.67082 秒目标还差 0.53077 秒。[验收、冷启动和显存](../../artifacts/video_runtime_v1/resident_route_v1/PROGRESS.zh.md)

### Risk 统计量按特征连续读取

三个稀疏档位默认启用。base 多视频按 5 个 prompt × 2 个 seed 验收，10 对中 9 对更快，最终 latent 均逐字节一致。GPU7 独立计时 base 5.17575 秒、Dense 6.54748 秒，当前 1.26503×。[本轮模块与视频验收](../../artifacts/video_runtime_v1/risk_variance_dot_v1/PROGRESS.zh.md)

### 稳定基数排序合并选页与复查标记

单视频三档共 14 对，13 对更快，最终 latent 均逐字节一致。默认 base 5.13247 秒，相对同卡原生 Dense 加速 1.27570×，距目标还差 0.45684 秒。[验收与回退](../../artifacts/video_runtime_v1/radix_select_v1/PROGRESS.zh.md)

### 重复文本交叉注意力

三档和 Dense 均默认启用匹配重复文本的 SM80 CUDA 路径。GPU7 最新 base 5.10156 秒、同主干 Dense 6.47587 秒，加速 1.26939×。距 1.40034× 验收线仍差 0.47706 秒。[模块、视频和缓存条件](../../artifacts/video_runtime_v1/repeated_context_v1/PROGRESS.zh.md)

### 融合注意力硬件计数

432 页真实输入上，融合核 Tensor Core 活跃率为峰值的 42.14%，HBM 吞吐为峰值的 10.41%。11 个边界、掩码、暂存及循环顺序候选均未通过模块速度验收。[原始计数及四种窗口对照](../../artifacts/video_runtime_v1/tail_boundary_v1/PROGRESS.zh.md)

### 精算概率成对转换

保留双项 AV 的 BF16 成对转换在模块跨卡复测中加速 1.0005–1.0141×。单视频三档和 base 的 5 prompt × 2 seed 共 28 对，latent 逐字节一致。多视频仅 5/10 更快，成对节省中位数 0.93 ms，保持现有部署。[模块与视频配对记录](../../artifacts/video_runtime_v1/packed_probability_v1/PROGRESS.zh.md)

### Risk 分块核的尾块计算量

288 页时将最后一块从 128 页缩至 32 页，模块加速 1.0762×，排序后的选页 ID 相同。base 单视频 4/8 更快，5 prompt × 2 seed 中 6/10 更快，分别节省 −0.51 ms 和 +1.12 ms（配对中位数），所有 latent 逐字节一致。端到端收益处于计时波动内，保留当前部署。[硬件计数、模块与视频](../../artifacts/video_runtime_v1/risk_remainder_v1/PROGRESS.zh.md)

### 精算注意力 softmax 尺度

保留双项 AV，将首页最大值用于后续精算页并在分母越界时恢复原始归约。通过溢出压力测试的核内版本在 base 单视频 5/5 对更快，配对节省中位数 34.29 ms（1.00675×），但新旧 latent NMSE 为 0.25197，最后一段达 0.70382。默认部署继续使用逐页归约，候选误差待用户判断。[实验报告和并排视频](../../artifacts/video_runtime_v1/softmax_scale_v1/PROGRESS.zh.md)

补充同一 Dense 参照的误差比较。city_walk / 701 整条 latent NMSE 从 0.51220 降至 0.49824（下降 2.73%），第 7 段改善最多、最后两段误差上升。[逐段误差与计算口径](../../artifacts/video_runtime_v1/softmax_scale_v1/PROGRESS.zh.md)

### 已接受的 base 首页缩放部署

用户接受首页缩放候选。base 默认开启，intersection、step 沿用逐页归约；base 可用 `--no-anchor-scale` 切回。正式实现四种窗口逐元素等于已接受原型，压力测试触发回退后等于原始在线归约。city_walk / 701 的正式入口输出与已接受候选逐字节一致，关闭后与原 base 逐字节一致。同卡 GPU7 各自预热后 5 次中位数，base 5.07839 s、原生 Dense 6.47279 s，加速 1.27458×。按 1.40034× 验收线仍需缩短 0.45609 s。[部署验收](../../artifacts/video_runtime_v1/softmax_scale_v1/PROGRESS.zh.md)

### 已接受的 base 分母列累加

正式入口在 GPU7 各预热后测 5 次，base 5.05312 秒，原生 Dense 6.46712 秒，加速 1.27983×。目标用时 4.61825 秒，还需缩短 0.43487 秒。正式输出与接受原型逐字节相同，关闭列累加后与上一部署逐字节相同。新时间线上融合核累计 812.52 ms。[模块、单视频与 Dense 误差](../../artifacts/video_runtime_v1/anchor_accum_v1/PROGRESS.zh.md)

### 补偿缩放验收与暂停点

单视频 3/8 对更快，配对节省中位数 -0.79 ms，加速比中位数 0.99984×；5 prompt × 2 seed 3/10 对更快，配对节省中位数 -1.77 ms，加速比中位数 0.99965×。本轮候选未通过视频速度验收，base 保留已接受的精算分母列累加。 同卡时间线显示融合核实测 808.72 ms，若按各段对应 Dense 的有效吞吐折算约为 528.33 ms。本轮结束按用户要求暂停，已验收部署仍为 1.27983×。[实验与逐段估算](../../artifacts/video_runtime_v1/compensation_scale_v1/PROGRESS.zh.md)

### 6135662 融合注意力评审与实验结论

评审定位到精算结果和补偿均值两个跨阶段存活的 128 维向量，提出用 `mx=le−4、den=1、am=exact` 接续状态，删除尾部独立合并。P432 完整融合核在 16 对交替测量中全部变慢，中位耗时从 1.14291 增至 1.17696 ms，加速比 0.97101×。spill 槽从 72 降至 26，输出 NMSE 为 1.08e−9。按评审约定在模块阶段结束此候选。随后验证精算页的 warp 分工和共享内存流水，获益的 K/V 调度见下节。[候选、原始计时和计数](../../artifacts/video_runtime_v1/shared_accumulator_v1/PROGRESS.zh.md) · [双项 AV 与理论时间口径修正](evidence/video_runtime_v1/PROFILE_MODULES_105.zh.md)

### 精算页 K/V 错开等待已在 base 启用

SM80 上保留 Q64 和双项 BF16 AV，提前下一页 K，推迟对 V 的等待。融合核设备累计从 808.57 降至 797.19 ms，距 Dense 吞吐折算参考仍有 268.86 ms。正常启动路径单视频 8 对全部提速，配对中位节省 11.02 ms。5 prompt × 2 seed 全部 latent 逐位相同。GPU7 正式入口 base 5.03268 s，原生 Dense 6.46920 s，加速 1.28544×。可用 `--no-page-prefetch` 回到上一版。P432 的短 scoreboard 等待升至 19.39%，下一轮可定位概率片段交换和 AV 等待的具体指令。[模块、硬件计数与视频验收](../../artifacts/video_runtime_v1/warp_rows_v1/PROGRESS.zh.md)
