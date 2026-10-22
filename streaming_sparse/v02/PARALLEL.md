# Full 多 GPU 流水线与评审交付

当前候选卡：物理 GPU 0–7。训练时每卡一项任务；训练完成后评估时每卡最多两项。空卡仅在显存低于 512 MiB 时接收首项任务；第二项只加入已确认属于本实验的 GPU 上下文，不终止外部任务。

## 工作顺序

1. 依据已完成的 Pilot 训练稳定性门槛决定是否扩大 Full；不使用 Test 分数作门槛。
2. 恢复缺失的 Pilot 评估，同时把剩余 12 个 Prompt 分为四组、分别采集；每组独立输出，避免多进程同时修改一个数据索引。
3. 合并数据时校验文件碰撞、完整事件笛卡尔积和 Prompt 划分；得到 6912 Events、1728 Snapshots。
4. 调度 5 个 Variant × 3 个 Router Seeds，共 15 组独立训练，各 3000 次更新。
5. 每组训练完成后产生 Train/Val/Test 三项评估依赖，共 45 次完整评估。优先出 Val/Test 结果，训练与独立评估可同时在不同 GPU 上运行。
6. 全部完成后重新汇总指标，运行完整性验收和机制分析，再自动打包。

`parallel_status.json` 包含任务队列、各 GPU 的活跃任务和失败记录；`logs/parallel_commands.jsonl` 保存命令、GPU、PID 和退出码。终止信号导致的单项任务最多重试两次；训练从持久优化器状态恢复，评估从已刷盘 Snapshot 恢复。Python/数值错误保留日志并停止相应依赖。

## 运行命令

在已有实验全部退出后，使用 RUNTIME.md 的 tmux 托管方式，将监督器命令替换为：

```bash
streaming_sparse/.venv/bin/python -u -m streaming_sparse.v02.supervisor \
  --gpu 7 --gpus 0 1 2 3 4 5 6 7 --max-restarts 2
```

有运行时不重复启动。`waiting_resources` 表示正在等卡，不能理解成实验完成。

## 自动验收与评审包

`review_pack.py` 在全部必需产物齐全后才会生成完成标记，验收包括：

- Full 数据的 6912 Events、Prompt 隔离划分、采集数值误差与完整覆盖。
- 15 组各 3000 次更新，主要模块梯度有限且具有有效梯度，best/final/latest 检查点存在。
- 45 次评估的逐 Snapshot、Step、Tile、Method、Keep 组合无缺失或重复。
- 每行历史 Token Budget、实际保留率、Prompt/Head/Layer/位置与数据索引一致。
- 从原始 SSE / Dense 能量重算全局和 Prompt/Head 主指标，与汇总一致。
- v0.1 的 40 个冻结文件及上游模型代码完整性。

最终交付：

- `streaming_sparse/artifacts/v02_review_latest.zip`：代码、配置、报告、三 Seed 消融、分组指标、Oracle 诊断、曲线、训练指标和最佳 Router 权重。
- `streaming_sparse/artifacts/v02_raw_results_latest.zip`：所有原始逐 Tile 结果，单独存放以便主包快速审阅。
- `streaming_sparse/artifacts/v02_REVIEW_COMPLETE.json`：完整验收数量、真实文件位置及 ZIP SHA-256。

每个 ZIP 内附成员清单和 SHA-256，并检查成员 CRC；大文件实际写到 `/tmp/jiangzhikun_self_forcing_sparse_v02/review/`，模型目录内保留链接。完成标记尚不存在时，表示尚未完成全部实验与打包。

`reports/FINAL_ANALYSIS.md` 将报告三训练 Seed 均值/标准差、按独立 Prompt 配对的置信区间、各 Head 相对 Mass 的误差，以及保持负 Recovery 的 Oracle Headroom 诊断。原始 QKV 和 Backbone 权重不打入评审包，保留数据索引与复现采集方式。

## 外部中断后的资源等待

`completion_watch.py --auto-resume --gpus 0 1 2 3 4 5 6 7` 独立观察完成标记与资源。外部 SIGTERM/SIGHUP 中断且本实验进程已全部退出时，仅在候选 GPU 空闲后恢复监督器；其他任务占卡时写入 `waiting_resources`，不争抢或终止外部进程。数值或 Python 错误停止并保留日志。显式 `supervisor.stop` 会阻止自动恢复。

续跑日志：`logs/completion_watch.log`；最近状态：`logs/completion_watch.json`。完整评审包通过校验后，入口写入 `reports/REVIEW_READY.md`。

## 训练完成后的评估并发

2026-10-16，15 组训练全部保存后，将 configs/evaluation_concurrency.json 设置为每卡最多两项独立评估。训练仍独占每卡。第二项评估仅在已有评估推进了 CUDA Snapshot、NVML 只有一个应用上下文且总显存小于 8 GiB 时加入；存在额外上下文时不追加。所有评估沿用同一 checkpoint、命令和数值路径，并从已提交 Snapshot 恢复。计划内控制器重启记录于 provenance/evaluation_concurrency_change.json。

## 持久评审交付

压缩在本地 scratch 中完成并检查 CRC，随后将两个 ZIP 原子复制到真实模型目录 streaming_sparse/review_packages/v02/，核对 SHA-256 后才发布完成标记。该目录不依赖 scratch 符号链接；包内含报告和配置快照，便于直接移走评审。
