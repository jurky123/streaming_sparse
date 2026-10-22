# 长时间运行与停止

本轮当前运行由 tmux session `self-forcing-sparse-v02` 托管；Full 在物理 GPU 0–7 上调度独立实验，各模型保留原学习率、每步 Snapshot 数和更新次数。不要在已有运行存活时重复启动。

## 查看

```bash
cat streaming_sparse/v02/MONITOR.md
cat streaming_sparse/v02/supervisor.json
tail -f streaming_sparse/v02/logs/supervisor.log
tail -f streaming_sparse/v02/logs/workflow.log
tmux list-panes -t self-forcing-sparse-v02 -F '#{pane_pid} #{pane_dead} #{pane_dead_status}'
```

`supervisor.json` 记录监督器最近一次启动或退出事件；实时 Snapshot 进度读取 `MONITOR.md` 或对应 `eval_*/progress.json`。不能只根据旧 `status.json` 的 running 字段判断存活。

## 显式停止

```bash
touch streaming_sparse/v02/supervisor.stop
```

监督器检测到停止标记后终止自己的工作流进程组，不会自动重启。也可停止本实验的 tmux session；不操作其他 session。再次启动前需明确移除本实验的停止标记，确认旧工作流及其 GPU 子进程已经退出。

## 持久启动命令

在 Self-Forcing 根目录执行，启动前确认不存在本实验活跃进程及同名 tmux session：

```bash
gcc -shared -fPIC -O2 -Wall -Wextra streaming_sparse/v02/signal_audit.c \
  -o streaming_sparse/v02/libsignal_audit.so
tmux new-session -d -s self-forcing-sparse-v02 -c "$PWD" \
  "$PWD/streaming_sparse/.venv/bin/python -u -m streaming_sparse.v02.supervisor --gpu 7 --gpus 0 1 2 3 4 5 6 7 --max-restarts 2 >> '$PWD/streaming_sparse/v02/logs/supervisor.log' 2>&1"
tmux set-option -t self-forcing-sparse-v02 remain-on-exit on
```

监督器的进程锁和工作流 PID 检查会阻止同类重复启动。仅信号中断且状态未变为失败/暂停/完成时允许有限重试；训练数值错误会停止并保留证据。

`logs/signal_senders.log` 由原生信号处理器在信号到达瞬间记录发送者及父进程的 PID、可执行程序和脚本路径；不记录任意参数或内联代码，也不忽略或拦截原信号处理逻辑。迁移后需按上面的 gcc 命令重编译小型审计库。

评估的 `resume.json` 是已刷盘 Snapshot 的提交记录。模型参数、数据或设置改变后拒绝混用旧结果；`--resume-existing` 仅用于严格验证并采用本次旧格式中断评估的完整前缀。不要手动编辑提交偏移。

故障分析与证据见 [PROCESS_INCIDENT.md](provenance/PROCESS_INCIDENT.md)。
