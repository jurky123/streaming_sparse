# Streaming KV Sparse Routing / SlimAV

## 最新源码与实验（Opt30–Opt32，2026-10-26）

本次更新包含至 Opt32 的路由、稀疏 attention、分组/聚合训练、动作选择与视频评测源码。
**本次不上传原始实验数据、视频、checkpoint 或运行日志。** 关键汇总数字见
[实验总结](docs/EXPERIMENT_SUMMARY.txt)，源码范围及外部依赖见
[源码发布说明](docs/SOURCE_RELEASE.txt)。

主要入口位于 `migration/unified_router/opt30`、`opt31`、`opt32`；本地状态通过
`opt30_status.sh`、`opt31_status.sh`、`opt32_status.sh` 查看。实验资产不在 Git 中，
新克隆不能直接恢复本机已经完成的训练。CUDA 后端依赖可通过
`bash migration/fetch_sparse_backend.sh` 获取。

以下内容描述仓库已有的 **v0.8 历史迁移快照**；其中已追踪的旧权重不是本次新上传的训练权重。
最新全层稀疏方法、实际速度与质量限制以以上实验总结为准。

## v0.8 历史快照

基于 Self-Forcing 1.3B 的离线路由与骨干局部 LoRA 共适应实验。

**本仓库按要求不包含数据、公共骨干权重或 Python 虚拟环境。** 包含基础代码、历史代码依赖、固定协议/Prompt/split、v0.8 校准及 Router warm-start、当前训练 checkpoint、来源审计和重建工具。

## 从这里开始

1. 阅读 **[重建与继续实验说明](REBUILD.md)**。
2. `bash migration/bootstrap.sh` 重建 Python 3.10 环境。
3. `streaming_sparse/.venv/bin/python migration/download_models.py --full` 下载并校验原 revision 的模型资产。
4. 在单 GPU 上重新采集 Replay；需要原始完整审计和后续 seed 校准时，再重新采集旧 QKV。
5. 验证数据身份，完成新机器审计后，选择严格恢复或明确记录的新数据继续训练。

## 当前研究范围

- 保留每个原始 Query 独立 Mask；历史 Page64，训练 Keep20%，当前段原可见 KV 全保留。
- 只有原编号层 **4/14/24 的全部12Heads**进行历史稀疏，其余27层Dense。
- 四个原生 diffusion steps 全执行；历史 context/cache-write 前向Dense。
- F/A/J × K/V 共六组、三个训练seed的计划尚未完成；当前迁移恢复点属于 **F-K / seed0**，具体 update 见 [inventory.json](migration/inventory.json)。
- 现有两视频 PSNR/LPIPS 属于 update620 的三层稀疏评估，见 [结果补充](migration/two_video_results/FOLLOWUP_REPORT.txt)。没有全30层稀疏结果。
- 当前算法实现用常规 masked SDPA，没有稀疏 Kernel；不能把逻辑连接删除率视为加速比。

## 目录

| 路径 | 内容 |
|---|---|
| `streaming_sparse/v08/` | v0.8 原始科学实现、执行优化器及配置 |
| `streaming_sparse/v08/runs/seed0/` | calibration、warmstart_K/V、F-K 的 checkpoint 与训练记录 |
| `streaming_sparse/v02` … `v07`、`capture/` 等 | 保留运行所需历史模块/配置；不含其数据或旧训练产物 |
| `migration/` | 下载、校验、重采集、继续训练工具及锁定依赖 |
| `migration/origin_reports/` | 原机器来源记录；不是新机器的活动审计或调度状态 |
| `migration/inventory.json` | 文件SHA256、源码身份、checkpoint恢复点 |
| `README_SELF_FORCING.md` | 原 Self-Forcing 项目说明 |

这是某次训练边界的静态快照，原机器后续继续训练不会自动同步到本仓库。

原项目许可证保留于 [LICENSE](LICENSE)。
