# Streaming Sparse 实验导航

本实验的统一入口。原始 Self-Forcing 代码和模型权重在仓库根目录；独立实验代码、数据与结果在 `streaming_sparse/` 下。

## 当前 v0.2

| 内容 | 入口 |
| --- | --- |
| 当前阶段（自动更新） | [STATUS.md](v02/STATUS.md) |
| 每 30 秒运行监控 | [MONITOR.md](v02/MONITOR.md) |
| 持久托管与显式停止 | [RUNTIME.md](v02/RUNTIME.md) |
| 多 GPU 队列及自动评审打包 | [PARALLEL.md](v02/PARALLEL.md) |
| 增量实验报告 | [REPORT.md](v02/REPORT.md) |
| 目录说明、日志查看方法 | [DIRECTORY.md](v02/DIRECTORY.md) |
| 实验协议与执行方式 | [README.md](v02/README.md) |
| 固定数据划分与训练规模 | [protocol.json](v02/configs/protocol.json) |
| 总流程日志 | [workflow.log](v02/logs/workflow.log) |
| 报告索引 | [reports/README.md](v02/reports/README.md) |

## 保留的 v0.1

| 内容 | 入口 |
| --- | --- |
| 第一版代码、使用说明 | [README.md](README.md) |
| 初始 P1 实验报告 | [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) |
| 4 Prompt × 2 Seed 验证报告 | [REPORT.md](experiments/validation_4p2s/REPORT.md) |
| 已打包的评审资料 | [评审 ZIP](artifacts/self_forcing_sparse_review_20261015.zip) |
| v0.1 源码校验清单 | [reference_v01_sha256.json](v02/provenance/reference_v01_sha256.json) |

## 代码与产物

- `router/`、`attention/`：可移植的基础 PyTorch Router 与 Attention Statistics。
- `capture/`、`integration/`：Self-Forcing 采集及推理适配。
- `train/`、`evaluation/`：v0.1 训练与评估入口。
- `v02/*.py`：本轮诊断、采集、模型变体、训练、评估和工作流。
- `data/`、`runs/`、`experiments/`：保留的 v0.1 数据、权重和结果。
- `v02/data/`、`v02/runs/`：v0.2 数据与各次实验产物。
- `artifacts/`：评审压缩包、Dense 与 Streaming 示例产物。

v0.2 的大体积 QKV 和 Pilot/Full 运行目录通过符号链接存放于 `/tmp/jiangzhikun_self_forcing_sparse_v02/`。备份和迁移时需要一起保存，详细位置见 [storage.json](v02/provenance/storage.json)。
