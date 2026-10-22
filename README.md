# Streaming KV Sparse Routing — v0.8 迁移快照

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
