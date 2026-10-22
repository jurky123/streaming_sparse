# v0.2 研究运行入口

本目录独立于 v0.1。原源码定义及 SHA-256 保存在 `reference_v01.zip` 与 `reference_v01_sha256.json`；原实验结果和权重继续保留在原路径。

## 当前运行

- `status.json` / `STATUS.md`：当前阶段。
- `workflow.pid`：当前工作流进程；启动时检查重复进程。
- `REPORT.md`：根据已落盘结果生成的增量报告，明确标记未完成部分。
- `protocol.json`：固定 Prompt 原文、划分、Seed、采集及训练规模。
- `commands.jsonl` / `workflow.log`：各次运行命令、恢复过程与阶段记录。
- `storage.json`：大体积文件的实际存储位置。

本机工作盘空间不足，`data/shared`、`runs/pilot`、`runs/full` 指向 `/tmp/jiangzhikun_self_forcing_sparse_v02/` 下的本地存储。迁移或长期保存时，应连同该目录中的 tensor、checkpoint 和逐 tile 结果一起保存。

## 执行顺序

P0 沿用原 4 Prompt × 2 Seed QKV，比较 λ=0/1e-4/1e-3；不重新采集。后续用该旧数据的验证结果选择正则权重，旧数据 Prompt 全部属于新 Train split。

Pilot：Train 0–7、Val 16–17、Test 20–21，共 12 个 Prompt，各 Seed 0/1；3 层 × 4 Heads × 3 种历史长度 × 4 步，共 3456 Events、864 Snapshots。每 Event 采样 8 个 16-token tiles，按时间轮转和 2×2 空间分层采样；同一历史状态的四步使用相同位置。

Pilot 先训练 Shared / Conditioned 两个模型（各 1500 更新、Router Seed 0），每次更新均衡抽取 4 个 Layer/Head Snapshot，各自复用 H 处理四个去噪事件。验证集选择最佳 checkpoint。训练数值稳定且 Loss 有下降后才进入 Full。

Full：Train 0–15、Val 16–19、Test 20–23，共 24 Prompt × 2 Generation Seeds，6912 Events。复用 Pilot 的实际 KV，仅补采其余 Prompt；Pilot 的 Val/Test 始终留在相同的 Full split。

Full 五种模型分别使用 Router Seeds 0/1/2、各 3000 更新：Shared、Conditioned、Spatial、Frozen Grouping、Independent Block MLP。Mean Q → Dot Product Decoder 保持原形，不自动启动 P4 增强。

## 指标

主指标 `macro_prompt_head_relative_error`：每个 Prompt 内，先对各 `(layer, head)` 分别累计所有 Q 的 SSE / Dense 能量，再等权平均 Layer/Head，最后平均独立 Prompt。这样高能量 Head 不会主导主指标。

同时保存全局原始 SSE / Dense 能量比、Prompt 内能量加权误差、逐 tile Mean/Median/P90/P95；每个 Layer/Head/Step/History/Prompt/Generation Seed 的指标均可查。早期 Pilot 首轮验证汇总曾使用 Prompt 内能量加权口径，原记录以 `metric_v1` 后缀保留，已由原始逐 tile 结果重算并校正 checkpoint 分数，训练目标不变，修订记录见 `metric_revision.json`。

训练轮次使用固定分层验证子集（Pilot 24、Full 48 个 Snapshot）选择；最终评估使用完整 Train/Val/Test。checkpoint 选择不使用 Test。

`B4` 是相同 Partition、相同名义 token 预算的 Mass；`B4_cost_capped` 使用 Ours 的实际 token 数作为预算上限，仍记录贪心未填满的差值。不同方法比较时需核对实际利用率。

Oracle 仅用于评估：前向 Greedy、Mass 初始化的有限轮增/删/交换及 Mass 自身取最优。该参考不是数学最优上界。只在 Headroom > max(1e-8, 5% E_Mass) 时解释 Recovery，负值保留。

## 复现命令

在 Self-Forcing 仓库根目录、CUDA PyTorch 环境中：

```bash
python -m pip install -r streaming_sparse/v02/requirements.txt
python -m streaming_sparse.v02.workflow --gpu 0 --through full
```

已有运行时直接查看状态；流程退出后，用同一命令跳过已完成阶段并从持久 checkpoint 续跑。GPU、磁盘不足或数值检查失败时会保存失败状态，不覆盖 v0.1。

仅重新生成已有结果报告：

```bash
python -m streaming_sparse.v02.analyze_p0
python -m streaming_sparse.v02.report
```

各 run 保留 `initial.pt`、`best.pt`、`final.pt`、可恢复优化器状态 `latest.pt`，训练与验证曲线、分模块梯度、修复比例、Assignment、Hard Mask、实际 token 预算及原始逐 tile 结果。
