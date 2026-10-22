# 第一轮实现与执行记录

日期：2026-10-15。代码位于 `Self-Forcing/streaming_sparse/`，原仓库受 Git 跟踪文件的 diff 为空。完整命令和设计说明见 [README.md](README.md)。

## 已下载

- 官方 Self-Forcing 源码 commit：`33593df3e81fa3ec10239271dd2c100facac6de1`。
- Wan-AI/Wan2.1-T2V-1.3B revision：`37ec512624d61f7aa208f7ea8140a131f93afc9a`，保存在 `../wan_models/Wan2.1-T2V-1.3B/`。
- gdhe17/Self-Forcing revision：`2f8b779212da279d212c22a509b66ad6552f350e`，DMD 权重保存在 `../checkpoints/self_forcing_dmd.pt`。
- 权重与基础模型合计约 22 GB。实验使用单张 H100 80GB，PyTorch 2.6 / CUDA 12.4，Backbone BF16，Router 和统计 FP32。

本机 Flash Attention 3 返回接口与官方代码不兼容，因此通过上下文适配器使用 SDPA。没有修改原模型源码。依赖版本见 `requirements-tested.txt`。

## 已实际执行

1. 官方 Dense Pipeline，1 个 prompt，6 latent frames，4 个去噪步骤，EMA checkpoint，输出视频和 latent。
2. 真实 QKV 采集：官方 prompt 文件前 2 个 prompt，layer 4 / head 0（均为从 0 开始编号），12 latent frames；3 种历史长度，全部 4 steps；每事件采样 4 个连续 Q 窗口，窗口最大 32 tokens。
3. 数值与梯度检查，包括真实 RoPE QKV、全保留重建、Hard Mask 直接 Attention 对齐、纯 Attention Loss 下各模块梯度、软预算隐式导数有限差分、带 sink 的窗口滚动和 H 复用。
4. Ours 与 B3 各 200 步，seed 0，学习率 1e-4，训练 Keep=0.3/0.2。训练和评估使用同一份 QKV；P1 两个 prompt 全部属于训练集。
5. 五组方法加 Random / 同分区 Mass，评估 Keep=0.3/0.2/0.1；结果逐 tile 记录，并按 prompt/layer/head/step/history length 汇总。
6. 训练后 Router 接入实际流式生成：1 个 prompt，6 latent frames，仅 layer 4 / head 0 稀疏，Keep=0.3；执行真实 KV Gather 与 Attention，保存视频。
7. Router 模块同步 profile。测量时 GPU 同时存在其他任务，耗时仅供定位瓶颈，不用于速度结论。

## 数值与缓存证据

- Query Events：24；实际内容去重后的历史快照：6。
- 历史 token 数：4680 / 9360 / 14040，每种长度 8 个 events。
- 全部捕获 events 的 FP32 Dense / backend 最大相对误差：3.0832e-06。
- 最终真实样本 Hard Statistics / 直接 Gather Attention 最大相对误差：3.36041e-13。
- Grouping、Content、Aggregator、Encoder、Decoder 的纯 Attention Loss 梯度均为非零且有限；200 步训练日志保留各模块梯度范数。
- 实际流式运行：1172 个 Query Tiles，共预计算 H 1 次；其余调用复用 H。
- 流式历史 KV 实际保留率范围：0.295513–0.300000。
- 窗口滚动 fixture 验证 sink 保留、被驱逐位置和当前 chunk 边界；两个非空历史版本分别只预计算一次 H，解析统计与实际稀疏适配器输出对齐。

## 固定训练样本的 P1 结果

以下是相同捕获 QKV 上的每个 Query Tile 相对误差的均值，包含 24 events × 7 tiles。末尾短 tile 按实际长度计算。属于训练集结果，不能推断跨 prompt 泛化。

| 方法 | Keep=0.3 | Keep=0.2 | Keep=0.1 |
| --- | ---: | ---: | ---: |
| B0 | 7.85499e-05 | 0.000182834 | 0.000656285 |
| B1 | 7.53394e-05 | 0.000174277 | 0.00043257 |
| B2 | 7.90677e-05 | 0.000176634 | 0.000456687 |
| B3 | 0.000134564 | 0.000268337 | 0.00142995 |
| Ours | 0.000128797 | 0.000237852 | 0.000619345 |
| Mass_same_partition | 9.67389e-05 | 0.000200701 | 0.000462975 |
| Random | 0.00850993 | 0.00940618 | 0.0125542 |

- Ours Keep=0.3：训练前 0.00624747 → 训练后 0.000128797，约降低 48.51 倍。
- Ours Keep=0.2：训练前 0.00822928 → 训练后 0.000237852，约降低 34.60 倍。

本次 P1 能证明真实 QKV → 可微分组 → Hard Mask Attention Loss → 参数更新的链路可学习。Ours 明显优于随机选择，并在本次均值上接近或优于 B3，但尚未优于 Mass。Keep=0.1 只用于压力评估，不在训练预算中。

各方法实际保留率并不完全相同。`per_tile.jsonl` 记录全部实际 token 数；同分区 Mass 以 Ours 实际 token 数为预算上限，但受贪心选择影响仍可能少保留，汇总另有完全相同 token 数的配对结果。不要仅看上表推断严格等成本收益。

## 可审查产物

- `artifacts/validation_synthetic.json`：预算、梯度、缓存滚动与稀疏适配器检查。
- `artifacts/validation_real.json`：4 个真实 events 的检查。
- `artifacts/validation_final.json`：批量统计实现的真实数据复核。
- `artifacts/dense_reference/`：官方 Dense 视频、latent、运行元数据。
- `data/p1/`：历史快照、Query Events、manifest 和 prompt split。
- `runs/p1/initial_router.pt`、`router.pt`、`latest.pt`：初始化、推理权重、可恢复训练状态。
- `runs/b3_p1/`：单独训练的 B3。
- `runs/p1/eval_initial/` 和 `eval_train/`：训练前后逐 tile 和分层汇总结果。
- `artifacts/streaming_p1/`：真实 Hard Mask 生成视频、latent 和 1172 条路由记录。
- `artifacts/router_profile.json`：独立模块成本，包含 Gather 和预计算摊销次数；当前是共享 GPU 条件。

## 尚未执行的正式研究

P2 的 64/16/16 prompt 划分、多个 layer/head、3 个随机种子、Query Tile 8/16/32 与 alpha 0/0.5/1 的比较均提供运行入口和配置，尚未执行。当前不能宣布在未见 prompt 上优于 Mass，也没有完整视频质量或加速结论。没有实现高性能稀疏 kernel；当前 P3 是用于数值与缓存验证的 PyTorch 参考路径。
