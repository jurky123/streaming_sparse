# 新机器重建与继续实验

## 0. 本仓库能恢复到什么程度

可以恢复代码、已训练 Router/LoRA、校准参数、优化器及采样器的保存状态。**没有历史 QKV、Replay latent/noisy states、教师预测 tensor、文本 embedding tensor、VAE/骨干权重或视频文件。** Prompt 文本与split属于固定协议，已保留。

不迁移数据意味着需要重新采集，不能凭最终 latent 猜造中间 noisy states。新数据指纹不同时，本轮从权重继续的结果应单独记为新分支，不能混入原六组配对实验。

当前 checkpoint 的组别、update、SHA256 以 `migration/inventory.json` 为准。`best.pt` 用于验证最优模型；`recovery.pt` 用于最近更新边界的恢复。不要把原机器 PID、GPU 编号、running 标记当作新机器运行状态。来源报告专门放在 `migration/origin_reports/`，没有放入活动 `v08/reports/`。

## 1. 目录与资源

为保持 checkpoint 的科学源码身份以及协议中的旧 QKV 绝对路径，优先使用原根目录：

```bash
mkdir -p /home/iflab/models
git clone https://github.com/jurky123/streaming_sparse.git /home/iflab/models/Self-Forcing
cd /home/iflab/models/Self-Forcing
```

如新机器用户名不同，可把实际工作目录通过兼容路径映射到 `/home/iflab/models/Self-Forcing`；需要有建立该路径的权限。**不要直接改 `v08/configs/protocol.json` 的路径再拿旧 checkpoint resume**：协议参与源码hash。必须改协议时应另外建立新实验身份。

原机器使用A100-SXM4-80GB。新机器显存与速度需要实际验证，尤其J组含激活反向图；本说明不保证小显存卡可运行。默认只用一张GPU，所有示例指定物理GPU0。

重新下载全部公共模型约21.8GiB；Replay及embedding约2.08GiB；完整旧QKV重采集约76GiB。若计划重建全部依赖，建议至少预留150GiB，另留Python环境及后续结果增长空间。不需旧QKV时可少用大量磁盘，但原始完整审计/后续seed校准目前仍依赖它。

## 2. Python与依赖

要求Linux、Python3.10、可运行CUDA12.4 wheel的NVIDIA驱动。通过本机网络直连，清除代理变量：

```bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
bash migration/bootstrap.sh
source streaming_sparse/.venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

固定PyTorch2.6.0+cu124、torchvision0.21.0；其余依赖来自当前机器锁定记录 `migration/requirements.lock.txt`。原始freeze另存于 `migration/environment.freeze.txt`，其中的旧绝对editable路径不能直接安装。FlashAttention不是必须项，现有capture支持SDPA。

`migration/verify.py` 只检查CPU上的文件、checkpoint和科学/执行源码身份，不调用GPU、不开始训练，也不证明新机器的数值等价性。下载或安装失败时保留错误记录，不静默替换核心版本。

## 3. 重新下载原 revision 模型

```bash
python migration/download_models.py --full
```

脚本通过本机直连下载，使用两个固定revision，并对下载文件逐个校验原记录SHA256：

- `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a`
- `gdhe17/Self-Forcing@2f8b779212da279d212c22a509b66ad6552f350e`

`--full` 包含T5、Tokenizer与VAE，供重新采集和视频解码使用。不加该选项只下载当前加载器需要的Wan基础diffusion模型、config及Self-Forcing checkpoint，不能直接重采集缺失embedding。

若官方端点不可达，可显式加 `--endpoint https://hf-mirror.com`；这与使用系统代理不同，revision和SHA校验不变。Self-Forcing checkpoint的SHA256必须是：

```text
a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3
```

## 4. 重采集数据：先Replay，再决定是否补旧QKV

### 4.1 Replay（主训练实际需要）

```bash
CUDA_VISIBLE_DEVICES=0 bash migration/recollect.sh replay
```

这会按仓库内已有Prompt registry，采集32Train/8Val Prompts ×2生成seed，每条轨迹覆盖三个历史长度与四个真实diffusion steps，共768Train、192Val完整当前段输入。输入包括实际历史latent、noisy latent和教师原生x0预测；没有从旧QKV替代Student路由。新的16个Test Prompts维持封存，不在该脚本中采集或评估。

重新采集后检查：

```bash
python migration/continue_training.py --check-only
```

若输出 `replay_identity_matches=false`，严格恢复会被拒绝。这可能来自重采集的数值/序列化身份变化；仍需核查原Prompt/split/seed、模型hash、native steps、history与可见性一致。不能删除指纹检查伪装成原样恢复。

### 4.2 旧全12-head QKV（原始Phase0、执行等价审计和seed1/2校准需要）

```bash
CUDA_VISIBLE_DEVICES=0 bash migration/recollect.sh legacy-qkv
```

按原24Prompt、2生成seed、层4/14/24、12Heads、12latent帧、8×16采样Query、SDPA重新采集到协议期望的 `v04/data/full`，并恢复原16/4/4 split。该数据约76GiB，只用于旧校准/诊断，不替代v0.8网络输出Replay。脚本拒绝向已有manifest追加一轮重复采集；若中断，应先审计已完成capture，不盲目重跑。

**没有迁移数据并不等于不再需要数据。** 若不愿重新采集旧QKV，原 `audit --real` 和 `execution.benchmark --full` 的原样入口会因缺依赖而停止；本仓库不通过复制旧passed报告绕过该门槛。已有seed0校准权重保留，但后续seed1/2的新校准仍需这些数据，或另立并审计新的Replay校准协议。

## 5. 新机器正确性审计

在重采集完成后，先验证全部当前文件，再在同一单卡顺序执行。以下命令有实际GPU计算，可能较慢：

```bash
python migration/verify.py
CUDA_VISIBLE_DEVICES=0 python -m streaming_sparse.v08.audit --tiny-network
CUDA_VISIBLE_DEVICES=0 python -m streaming_sparse.v08.audit --real
CUDA_VISIBLE_DEVICES=0 python -m streaming_sparse.v08.execution.benchmark --full
```

后两项仍读取旧QKV。新的活动报告写到 `streaming_sparse/v08/reports/`；来源报告始终保留在 `migration/origin_reports/`。原科学源码hash、执行优化hash和预算/梯度容差没有改变。不要因新硬件误差而事后放宽阈值。

可选CPU语法检查：

```bash
python -m compileall -q streaming_sparse model pipeline utils wan migration
```

## 6. 从当前训练权重继续

先确保没有其他本仓库训练进程使用同一输出目录。不要启动旧workflow状态副本，不要复制旧PID。

### 数据指纹完全一致：严格resume

```bash
CUDA_VISIBLE_DEVICES=0 python -u migration/continue_training.py
```

入口校验Train/Val指纹、审计与源码身份，然后通过原执行worker加载 `recovery.pt` 的Router、LoRA、优化器和采样器状态，继续剩余update预算。原训练Dropout=0；更换硬件后仍不承诺逐bit重现。

### 数据指纹不同：明确的新数据继续分支

```bash
CUDA_VISIBLE_DEVICES=0 python -u migration/continue_training.py --new-replay
```

此入口不会伪造原数据指纹。它归档原F-K目录到 `migration/continuations/<UTC>/origin_run/`，从相同Router/LoRA权重开始，**重置AdamW状态、采样覆盖率、验证best/curve及update为0**，执行新的2000updates。父checkpoint SHA、父update、新数据指纹和重置行为写入 `BRANCH.json`，当前run也带 `CONTINUATION_BRANCH.json`。

该分支只用于显式的迁移后继续实验，**不是原六组配对主实验的等价续跑**。正式补齐六组×三个seed时，应使用同一套重建数据、配对初始化和一致训练预算，建立新的完整run归档；不能将新F-K与旧其他组混算。不要直接启动矩阵launcher，让它把这个单独分支误当作原任务完成。

后续seed1/2需要各自重新校准与warm-start；不要把seed0的校准冒充另外两个独立初始化种子。所有结构与checkpoint选择仍仅用Val。

## 7. 监控、评估与已知边界

直接训练的stdout会输出更新与验证信息；另可查看当前目录：

```bash
tail -n 5 streaming_sparse/v08/runs/seed0/F-K/metrics.jsonl
cat streaming_sparse/v08/runs/seed0/F-K/recovery.json
cat streaming_sparse/v08/runs/seed0/F-K/validation_curve.json
nvidia-smi
```

`status.sh`面向原workflow调度器；单独运行迁移入口时，应以当前worker日志/metrics/recovery为准。不要把来源报告中的running误当作新机器任务。

按原evaluate入口做未见Prompt误差评估，务必给新输出目录；不要覆盖来源结果：

```bash
CUDA_VISIBLE_DEVICES=0 python -m streaming_sparse.v08.launcher.worker streaming_sparse.v08.evaluate \
  --checkpoint streaming_sparse/v08/runs/seed0/F-K/best.pt --split val \
  --output streaming_sparse/v08/evaluations/migrated_val --mass-reference
```

PSNR/LPIPS评估需要VAE及AlexNet的LPIPS预训练权重；LPIPS0.1.4已列入环境锁。原AlexNet缓存不在仓库，需要在新机器直连重新下载。原两视频质量/计时记录属于update620、仅三层稀疏，不能称为全层稀疏或真实视频质量保证。

## 8. 静态快照与上传验证

`migration/inventory.json` 列出此次迁移文件的大小与SHA256，明确 `data_included=false`。仓库没有原Python环境或任何数据Tensor；模型资产下载脚本不是实验数据恢复脚本。

`migration/VALIDATION.txt` 只记录在来源机器上做过的CPU文件/源码/checkpoint与脚本检查。**尚未在另一台机器完成安装、重新采集、GPU审计或恢复训练**，不得把这些未执行步骤标成通过。

## 9. 本机已恢复后的八卡运行（2026-10-22）

本机的模型、768 Train / 192 Val Replay、完整旧 QKV、原科学审计和执行审计均已恢复并重新验证。当前是迁移后的 F-K/seed0 新数据继续分支，父 checkpoint 为原 update640；原 run 已归档。用户进一步明确授权“允许batch8”，有效配置为八张 A100、每卡一个完整输入、全局 batch8、2000 updates、学习率不变。对应总输入数16000，不能与原 batch4 的配对主实验混算。授权记录在 `migration/restoration/20261022_133350/batch8_amendment.json`，活动 run 也有 `BATCH8_AMENDMENT.json`。

执行适配器位于 `migration/runtime/`，原科学源码和 `v08/execution/` 保持原身份。Router score 批次128→512、连续 gate VJP 批次128→256，原64次预算求解使用 CUDA Graph；逐 query 路由、辅助样本、精度和原审计容差不变。数据并行汇总 objective/8 的梯度，再按原规则裁剪和执行 AdamW；仅 rank0 写 checkpoint，每次完整更新保存 recovery。验证按轨迹分片后按原顺序合并。完整192输入验证与单卡误差逐输入完全一致。

长时间训练使用独立 tmux 后台会话启动或从最新 recovery 继续，避免交互命令会话结束影响任务（已有后台会话时不重复启动）：

```bash
./start_training.sh
```

后台会话名为 `cavir-batch8`，使用专用 tmux socket `streaming_sparse`；`./progress.sh` 仍用于查看进度。需要查看后台控制台时，运行 `tmux -L streaming_sparse attach -t cavir-batch8`，按 Ctrl+B 后按 D 脱离会话。

原恢复入口也保留，适合已处于持久终端会话内时使用：

```bash
PYTHONPATH=. streaming_sparse/.venv/bin/python migration/run_batch8.py
```

只检查恢复点：

```bash
PYTHONPATH=. streaming_sparse/.venv/bin/python migration/run_batch8.py --check-only
```

不要对这个分支再次运行 `continue_training.py --new-replay`，否则会另建分支并重置优化器。当前实时状态、逐卡日志及实测报告：

```text
migration/restoration/20261022_133350/continuation_status.json
migration/restoration/20261022_133350/batch8_training/launcher_status.json
migration/restoration/20261022_133350/batch8_training/rank0.log
migration/restoration/20261022_133350/performance8.json
streaming_sparse/v08/runs/seed0/F-K/metrics.jsonl
streaming_sparse/v08/runs/seed0/F-K/recovery.json
```

适配器启动前要求本机的 Router 完整输入等价审计、对应 world size / batch 的梯度和 AdamW 审计全部通过；运行中检查适配器 hash。需要进一步修改 `migration/runtime/*.py` 时，先在更新边界保存并停止训练，再修改、审计和恢复。

八卡适配器不安装原单卡 `RecoveryAdamW` hook：该 hook 的闭包会绕过后置 rank 判断，在第10步引起多个进程写同一个临时文件。现在保持原生 AdamW，仅由 rank0 保存；八进程20步保存回归测试已通过。首次运行的 update10 ZIP 目录已按本地记录的 CRC 修复，2050个张量存储不变，505项优化器状态和80次采样均已校验，从update11续跑。该步未能输出的 loss 不补造，记录在活动 run 的 `metrics_gaps.json`；事件与恢复校验保存在 `migration/restoration/20261022_133350/checkpoint_race_incident/`。

可读进度面板：运行仓库根目录的 `./progress.sh`，每5秒自动刷新训练进度、预计剩余时间、loss趋势、验证结果、checkpoint及八卡占用；Ctrl+C退出。`./progress.sh --once` 查看单次快照。该入口仅使用系统Python标准库和nvidia-smi，只读文件，位于训练身份目录之外。

### R1 训练优化（2026-10-23）

第819次更新为冻结对照点，保存于 `migration/routerslim_runs/20261023_035515/baseline.pt`。Profile、逐项消融、完整网络输出/梯度/AdamW检查与说明见该目录的 `REPORT.md`、`baseline_profile/profile_breakdown.csv` 和 `deployment_audit.json`。

新代码放在 `migration/routerslim/`，不改科学源码和原 `migration/runtime/` 适配器。只有验收通过并生成 `migration/routerslim_deployment.json` 后，原 `./start_training.sh` 才会选择 R1 入口；入口校验代码、证据 SHA，并从原 optimizer/sampler 状态继续。当前实际启用方差等权预聚合和权重为零时的 Attention 辅助分支停算。内容投影前移通过正确性检查但训练实测较慢，默认关闭。Soft Gate、校准辅助项、完整残差范数、历史保留率与2000更新预算保持原协议。
