# Streaming KV Interaction-aware Sparse Attention v0.1

4 Prompt × 2 生成 Seed 的独立训练/验证实验已完成，见 [验证报告](experiments/validation_4p2s/REPORT.md)；精确 Prompt 划分、Seed 与配置记录在同目录。采集器支持 `--seeds 0 1`，为每个 Prompt 使用相同的两个 Seed。

实验代码全部放在本目录；不修改 Self-Forcing 的任何受 Git 跟踪文件。`router/` 和 `attention/` 只依赖 PyTorch，模型专属逻辑集中在 `capture/` 和 `integration/`。从 Self-Forcing 仓库根目录运行以下命令。

## 环境和模型

本机解释器：`streaming_sparse/.venv/bin/python`。该虚拟环境复用本机现有 PyTorch 2.6 CUDA 12.4，单卡运行。移植时用自己的 CUDA PyTorch 环境安装 `pip install -r streaming_sparse/requirements.txt`，无需移植 `.venv/`。

官方代码：<https://github.com/guandeh17/Self-Forcing>。当前源码 commit 为 `33593df3e81fa3ec10239271dd2c100facac6de1`。

权重已按官方地址下载到仓库的 `wan_models/Wan2.1-T2V-1.3B/` 和 `checkpoints/self_forcing_dmd.pt`。下载脚本和精确 Hugging Face revision 分别在 `scripts/download_assets.py` 和 `artifacts/model_revisions.json`。下载脚本仅读取进程代理环境，代码中不保存凭据。

```bash
export PYTHON=streaming_sparse/.venv/bin/python
# 移植后重新下载（使用自己的已授权网络设置）：
$PYTHON streaming_sparse/scripts/download_assets.py
```

## P0：官方 Dense 推理与真实 QKV

```bash
$PYTHON -m streaming_sparse.capture.capture_runner \
  --mode dense --frames 6 --limit 1 --backend sdpa --save-video \
  --output streaming_sparse/artifacts/dense_reference

$PYTHON -m streaming_sparse.evaluation.validate --cache-check \
  --output streaming_sparse/artifacts/validation_synthetic.json

# P1 采集两个 prompt，一个代表 layer/head。12 latent frames 覆盖 1/2/3 个历史 chunk。
# 采集 32 连续 Q 的窗口，之后可从同一数据安全切分成 8/16/32 的 tile。
$PYTHON -m streaming_sparse.capture.capture_runner \
  --mode capture --frames 12 --limit 2 --layers 4 --heads 0 \
  --query-tile 32 --tiles-per-event 4 --backend sdpa \
  --output streaming_sparse/data/p1

$PYTHON -m streaming_sparse.evaluation.validate \
  --data streaming_sparse/data/p1 --max-events 4 \
  --output streaming_sparse/artifacts/validation_real.json
```

使用官方 `CausalInferencePipeline.inference()`、原始 EMA checkpoint、原始 480×832 分辨率、4-step 时间表和 clean-context 更新。`--backend sdpa` 仅在上下文内选择 PyTorch SDPA；它处理 Cross-Attention 的 key length，不改变模型权重。当前机器安装的 Flash Attention 3 接口与官方调用签名不匹配，因此本轮使用 SDPA。移植时可安装兼容 Flash Attention 后使用 `--backend auto`。

捕获点是官方 `CausalWanSelfAttention.forward()` 内**实际调用 Attention 的位置**，而非 Q/K 投影 hook。原始投影、RoPE、KV 写入、滚动和输出投影全部执行；只临时拦截此模块的 Attention 函数。文本 Cross-Attention 不参与捕获或路由。当前仅支持 batch size 1，禁止同时安装两个 adapter。

`history_tokens` 从实际工作缓存的 `local_start_index` 和 Attention 可见切片共同计算。当前 chunk 全部 KV 始终保留。记录绝对 `(latent_frame, patch_y, patch_x)` 位置，包含 sink 与滚动后的不连续位置。clean-context 前向不作为训练 Query Event。历史快照使用实际 K/V、位置和 prompt/layer/head 的内容散列作为版本，跨去噪步骤去重。

保存结构：

```text
data/p1/
  history/<cache_version>.pt       # k_hist, v_hist, token_positions
  queries/<event_id>.pt            # q, all current KV, o_dense, o_backend, metadata
  events.jsonl                    # 通过 cache_version 关联
  splits.json                     # prompt ID 划分
  run.json                        # seed、真实时间步、设备、运行成本
```

Q/K/V 按真实 BF16 数据保存，Dense FP32 重算输出与原始 backend 输出分别保存。当前 KV 随 step 改变，必须随 event 保存。默认每事件只采样少量分散的连续 Q 窗口以控制磁盘；`--tiles-per-event 0` 捕获全部 Q。离线评估禁止把不相邻窗口拼成同一 tile。

## P1：完整 Router 过拟合

```bash
$PYTHON -m streaming_sparse.capture.dataset_io streaming_sparse/data/p1 \
  --train 2 --val 0 --test 0

$PYTHON -m streaming_sparse.train.train_router \
  --data streaming_sparse/data/p1 --config streaming_sparse/train/configs/p1.json \
  --output streaming_sparse/runs/p1

$PYTHON -m streaming_sparse.evaluation.eval_attention \
  --data streaming_sparse/data/p1 --checkpoint streaming_sparse/runs/p1/router.pt \
  --split train --keep 0.3 0.2 0.1 --output streaming_sparse/runs/p1/eval_train
```

本机还用 `train/configs/b3_p1.json` 训练了同样 200 步的 B3；评估时可追加 `--b3-checkpoint streaming_sparse/runs/b3_p1/router.pt`。每次新训练同时保存 `initial_router.pt`，可以在完全相同的 QKV 上比较训练前后结果。

P1 训练集评估只用于检查能否过拟合，不支持泛化结论。`metrics.jsonl` 记录真实 Hard Loss、Soft Loss、差值、实际保留率、block 大小范围及各模块梯度范数。首步对 Grouping/Content/Aggregator/Encoder/Decoder 的非零有限梯度做检查。数值验证另外仅使用 Attention Loss 检查梯度，避免正则项掩盖梯度断裂。

模型配置严格对应方案：Grouping 256→128→128→64；独立 Content 256→256→128；3 轮 Soft K-means；Hard Assignment STE；Mean+Hard Max；元信息含 log block size 和时空位置；2 层 Pre-LN Transformer（256/4 heads/1024 FFN）；Query 128→256→256 点积 decoder。Head/Layer Embedding 共享网络，分组和 mask 按单头执行。

聚类以固定均匀 token 索引初始化中心，给定输入和参数时确定。仅修复空 cluster 和超过 `ceil(max_size_factor*N/B)` 的异常大 cluster，不要求等大小。修复后每个历史 token 恰好属于一个 block；反向使用原始 soft assignment。使用轻量 soft usage KL 正则（`collapse_weight` 可设 0）；Mean 和 Attention Statistics 保留 assignment 梯度，Hard Max 仅向被选 token content 传梯度。Empty mask 保留在 Encoder 中，防止后续扩展 padding 时误用。

训练前向为真实 Hard Clustering + 真实 Budgeted Ranking。预算按历史 KV token 数计算，超预算候选跳过。软 gate 的阈值用二分求解，反向通过 token 预算等式做隐式求导；`validate` 含 score 和 cost 的有限差分检查。空预算不选历史，满预算保留全部；当前 chunk 永远保留。

统计计算使用 FP32、历史和当前共享 stabilizer、分块 QK，两遍计算不物化完整长序列 Attention Matrix。`U` 按小 Query Tile 计算，保留对 STE assignment 的梯度。Router 参数与优化器使用 FP32；真实 Backbone 与保存的 QKV 使用 BF16。第一版未开启 Router autocast，以便检查统计和梯度。

每个优化步骤按 history snapshot 分组，跨多个 diffusion steps 复用一次 H；参数更新后重算 H。`latest.pt` 含优化器和随机数状态，支持 `--resume`；`router.pt` 只含可移植 Router 配置和权重。

## P2：正式跨 Prompt 比较

```bash
$PYTHON -m streaming_sparse.capture.capture_runner \
  --mode capture --frames 12 --limit 96 --layers 4 14 24 --heads 0 3 6 9 \
  --query-tile 32 --tiles-per-event 4 --backend sdpa --output streaming_sparse/data/p2

$PYTHON -m streaming_sparse.capture.dataset_io streaming_sparse/data/p2 \
  --train 64 --val 16 --test 16 --seed 0

$PYTHON -m streaming_sparse.scripts.run_matrix \
  --data streaming_sparse/data/p2 --output streaming_sparse/runs/p2 \
  --seeds 0 1 2 --tiles 16 --alphas 0

# B3 必须单独训练固定空间分组 Router，不能用 Ours 冒充。
$PYTHON -m streaming_sparse.train.train_router \
  --data streaming_sparse/data/p2 --config streaming_sparse/train/configs/b3.json \
  --output streaming_sparse/runs/b3

$PYTHON -m streaming_sparse.evaluation.eval_attention \
  --data streaming_sparse/data/p2 \
  --checkpoint streaming_sparse/runs/p2/seed0_tile16_alpha0.0/router.pt \
  --b3-checkpoint streaming_sparse/runs/b3/router.pt \
  --split test --keep 0.3 0.2 0.1 --output streaming_sparse/runs/p2/eval_test
```

所有 split 按 prompt 文本散列划分，重复 prompt 不会跨 split；检测重叠立即报错。P2 采集可能较大，请按磁盘容量分批安排。采集支持 `--offset`；同一目录重复完全相同事件不会重复写入。最终 checkpoint 应使用 Val 结果选择，Test 仅用于最终报告；当前训练不自动挑选最佳 checkpoint。

五组基线已实现：B0 固定时空 block+Mass，B1 原始 Key 聚类+Mass，B2 原始 Joint KV 聚类+Mass，B3 固定时空 block+单独训练 Router，Ours 学习 Joint KV 分组+Router。额外提供 Random 和同一 learned partition、以 Ours 实际 token 数为预算上限的 Mass 对照。不同 block 大小导致预算可能未填满，逐 tile 报告 token 差值，不能忽略利用率差异直接宣称优于 Mass。Mass 使用每个 query 归一化后的精确 attention mass。

`per_tile.jsonl` 是可复查原始结果，`summary.json` 含 Mean/Median/P90、prompt 均值方差、按 prompt bootstrap 的 95% CI，以及 layer/head/step/history length 分层统计。可用 `--oracle-max-blocks 32` 在少量小分区上追加 Greedy 参考；它不是数学保证的性能上界，不进入训练。

Query tile 与成本优先级消融可用 `run_matrix --tiles 8 16 32 --alphas 0 0.5 1`。建议先跑默认组合再扩展。B3 使用同样 seed、训练步数和预算设置进行公平比较。

## P3：真实 Hard Mask 流式推理与成本

```bash
$PYTHON -m streaming_sparse.integration.streaming_runner \
  --mode sparse --router-checkpoint streaming_sparse/runs/p1/router.pt \
  --frames 12 --limit 1 --layers 4 --heads 0 --keep 0.3 --backend sdpa \
  --output streaming_sparse/artifacts/streaming_p1 --save-video

$PYTHON -m streaming_sparse.evaluation.profile_router \
  --data streaming_sparse/data/p1 --checkpoint streaming_sparse/runs/p1/router.pt \
  --output streaming_sparse/artifacts/router_profile.json
```

实际 sparse 路径先计算 block mask，再 Gather 原始 K/V，执行真正的 SDPA/兼容 Flash Attention。选定 layer/head 以外仍走 dense。H 在同一层/头的稳定历史版本内跨 Q tiles 和 diffusion steps 复用；新 chunk、窗口滚动或新 prompt 时重建。clean-context 更新使用 Dense Attention。当前实现用 Python 循环、单头和小 tile，定位为正确性参考，不能据此宣称内核加速。

`profile_router` 同步 CUDA 后分别测量 Feature、Cluster、Aggregation、Encoder、完整预计算、在线 route、Gather+Attention 和 Dense Attention。生成流程的 wall time（包含加载后的生成/解码/捕获）在 run.json 中。流式 metrics 的 `precompute_host_ms` 仅为主机提交耗时，不能当 GPU 预计算耗时；真实性能以同步 profile 为准。总成本应按真实复用次数 R 摊销，并计入 Gather 与所有未稀疏层/头的成本。本机初次 profile 时 GPU 同时有其他任务，记录仅供定位瓶颈，正式速度实验需独占 GPU 重新测量。在线推理不执行训练用的软阈值二分。

长视频缓存滚动可用 `--local-attn-frames` 与 `--sink-frames`，当前 QKV 捕获适配器已覆盖官方滚动分支；默认 global cache 仅允许最多 21 latent frames。

## 文件职责

| 目录 | 职责 |
| --- | --- |
| `capture/` | 官方 Dense 运行、真实 Attention 捕获、快照和 Query Event 存储、prompt split |
| `router/` | 可移植 PyTorch 网络、可微聚类、STE token 预算选择 |
| `attention/` | 精确可微 block statistics 与相对 Attention MSE |
| `train/` | 离线 Router 联合训练、P1/P2/B3 配置 |
| `evaluation/` | P0 验证、五组基线、分层评估、同步性能测量 |
| `integration/` | 恢复式 Self-Forcing 插桩和真实 Gather Attention |
| `scripts/` | 权重下载、可复现多 seed/超参数运行 |

审查重点：`evaluation/validate.py` 将检查项转为可执行断言；`artifacts/` 保存本机实际运行证据。未执行的 P2/P3 正式质量实验不得据代码存在而标记完成。
