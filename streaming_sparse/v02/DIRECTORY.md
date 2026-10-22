# v0.2 目录与查看入口

## 分类

```text
v02/
├── *.py                  实验实现，保持原模块导入方式
├── README.md             协议、训练和复现说明
├── STATUS.md/status.json 工作流自动更新的阶段状态
├── REPORT.md/results.json 自动生成的增量结果
├── configs/              固定协议、采集 Prompt、已选择的正则项
├── logs/                 总流程、采集、P0 和报告日志及命令记录
├── reports/              报告索引及快捷链接
├── provenance/           v0.1 快照、校验和、验收、指标修订、存储说明
├── data/
│   ├── shared/           真实 QKV，链接至本地 /tmp
│   ├── pilot/            Pilot 索引、Prompt 划分、覆盖统计
│   └── full/             Full 索引，完成采集后生成
└── runs/
    ├── implementation_check/ 早期实现验收产物
    ├── p0/               正则权重诊断及其报告
    ├── pilot/            链接至 /tmp 下的 Pilot 运行目录
    └── full/             链接至 /tmp 下的 Full 运行目录
```

已归档的根目录文件保留相对符号链接，因此原日志命令和运行中的程序仍可使用原路径。自动生成的根目录状态、报告，以及每个 run 的内部布局保持稳定；`reports/` 提供统一查看入口。后续 Full 阶段新生成的采集日志和 Prompt 文件可能仍出现在根目录。

## 每次训练的布局

`runs/{pilot,full}/{variant}/seed{0,1,2}/` 包含：

- `train.log`：终端训练日志；`metrics.jsonl`：逐更新损失、梯度与事件指标。
- `config.json`、`requested_config.json`：完整实际/请求训练配置。
- `initial.pt`、`best.pt`、`final.pt`：初始、验证集最佳及最终权重。
- `latest.pt`：优化器动量和随机状态，可恢复训练。
- `validation_curve.jsonl`、`learning_curves.png`：验证曲线与图像。
- `validation/step*/`：模型选择用的固定验证子集结果。
- `eval_train/`、`eval_val/`、`eval_test/`：完整数据评估；包含逐 Tile 结果和分组汇总。
- `eval_*/progress.json`：评估进度。对应 `summary.json` 写出后该次评估才完成。
- `assignments/`：诊断采样的分区和 Hard Mask。

`metric_v1` 后缀为指标口径修订前的保留记录；当前主指标为 `macro_prompt_head_relative_error`，详见 [修订记录](provenance/metric_revision.json)。

## 日志查看

从 Self-Forcing 根目录执行：

```bash
cat streaming_sparse/v02/STATUS.md
tail -f streaming_sparse/v02/logs/workflow.log
tail -f streaming_sparse/v02/runs/pilot/shared/seed0/train.log
cat streaming_sparse/v02/runs/pilot/shared/seed0/eval_train/progress.json
```

总流程日志跨恢复过程追加，含早期失败尝试。判断当前状态时应结合 `status.json`、`workflow.pid`、活跃进程和最近阶段日志。

## 存储与归档

大体积产物实际位于 `/tmp/jiangzhikun_self_forcing_sparse_v02/`，当前约 14 GB；主目录所在共享盘剩余约 11 GB（整理时快照）。代码、P0 产物、数据索引和报告保存在模型目录内。迁移时必须同时保存 `/tmp` 下的真实数据与 checkpoint，复制符号链接本身不足以备份数据。

整理记录：[directory_organization.json](provenance/directory_organization.json)。本次整理核对了 v0.1 的 40 个冻结文件及原始模型 Git 差异。

## 共享盘写满后的存储更新

运行日志、配置、状态、报告及所有大体积实验产物已转入本地 /tmp/jiangzhikun_self_forcing_sparse_v02/，旧路径保留链接。源代码仍在模型目录；官方 DMD 权重已校验 SHA-256 后移到本地并保留原路径链接。provenance/storage.json、storage_recovery.json 和 checkpoint_storage.json 记录详细位置。迁移时需保存整个 scratch 目录，不能只复制符号链接。
