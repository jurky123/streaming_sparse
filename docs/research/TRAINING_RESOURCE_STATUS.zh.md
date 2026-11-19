# 全覆盖 C 训练资源与状态记录

采样时间：2026-11-16T13:59:24.768980+00:00。这是当前 C 联合训练的 ETA，不包括随后 Full_Pset 的全覆盖适配。

| Seed | 进度 | 完成比例 | 最近墙钟秒/步 | 纯训练剩余 |
|---|---:|---:|---:|---:|
| 17 | 37,056/66,816 (55.5%) | 0.0844 s/step | 41.9 分钟 |
| 29 | 39,936/66,816 (59.8%) | 0.0723 s/step | 32.4 分钟 |

截至 2026-11-16T14:07:53.504561+00:00，当前估计训练剩余 41.9 分钟；加上最终验证与导出约 47–57 分钟。验证并行时可能发生短期 IO 竞争，ETA 使用最新实际墙钟吞吐。实际以 status 输出为准；若再次失败，ETA 会标记无效。

## 已落实的改动

- 每个事件 Risk 只计算一次，复用相同 FP32 特征与原候选规则。
- mmap 不再实体读取未使用的 page_output 教师；双线程、有界八事件锁页内存预取，按原顺序交付，异步传输。
- 冻结检查从每个冻结项重新构建整个 state_dict，改为一次构建再逐项比较。一次耗时从 10.507 s 降至 0.050 s；检查内容未删减。
- GPU0/1 分别继续 seed17/29 的串行 Adam 序列；GPU2–7 已完成 8,352 个冻结 OldRisk 基线事件的六路缓存，随后每个 seed 使用三张空闲卡对不可变 epoch 快照分片验证。下一 epoch 的训练不等待验证，最后收齐原顺序结果再导出。
- 最近墙钟吞吐包含检查点和校验开销；旧实现约 0.28–0.31 s/step，当前约 0.07 s/step，约四倍。只计算训练内核时间会低估 ETA，现已修正。

这些是执行组织优化，样本顺序、两次完整 epoch、学习率、Adam 状态和梯度精度均保留。普通有限输入的 12 步单视频回放中，模型、Adam、RNG 和损失记录逐位一致。验证分片属于原定训练工作，不是新增多视频加速测试。

## 顺带修复的原始数值问题

原始执行重放也在 step 23491、`data/toy_train_s708/c8_l28_s2_h11_t6.pt` 失败：主导质量直接相减后，512 个真实动作标签中有 16 个 NaN。这里只对非有限结果使用等价的正质量 log 混合重算，496 个有限项逐位保留；修复值与独立 CPU 参考最大差异 0。完整梯度更新已验证有限；没有跳样本、裁剪标签或更改预算。

原始 checkpoint 和两次接续前的备份均保留于 `execution_optimization/`。训练器现在明确写入 failed 状态和失败事件，避免进程停止后仍显示旧 training 状态。

## 资源边界

最近采样 GPU0/1 的平均利用率约 23–24%，显存不足 1 GiB；还没有吃满 A100。当前每次更新只有一个 head/event，且共享 Reader 的有序 Adam 更新构成串行依赖。直接改为八卡 DDP 或各卡独立 Reader 会改变有效 batch/更新轨迹，不能当作本次无损接续。六张卡在冻结基线完成后会间歇空闲，epoch 验证时再使用；不会把“八卡常驻”误报成“八卡满载训练”。

刷新状态：

```bash
/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.full_coverage.status
```

## 已完成的首轮原定验证

- seed17：8,352 个开发事件，C+Risk attention NMSE / 原基线 = 0.975366，下降 2.46%。
- seed29：8,352 个开发事件，C+Risk attention NMSE / 原基线 = 0.973102，下降 2.69%。

这些是固定事件 Attention NMSE，不是最终视频质量。当前预计时长只覆盖两份全覆盖 C 联合权重的训练、验证与导出；后续 Full_Pset 适配不在此 ETA 内。

## 下一阶段资源分配（用户已确认）

下一阶段 Full_Pset 适配采用四卡同步数据并行：seed17 使用 GPU0–1，seed29 使用 GPU2–3；GPU4–5 用于验证与缓存，GPU6–7 用于视频与运行时。当前 C 训练保持现有方式完成，不中途改变 Adam 更新序列。新阶段需记录有效 batch 与优化器更新次数。加速测试继续限定单视频，优先单模块隔离。分工文件：`local_experiments/full_coverage/next_stage_resources.json`。

## 2026-11-16 梯度失败修复

seed17 在尝试 update 48514、seed29 在尝试 update 58929 时触发非有限梯度检查；最近安全 checkpoint 分别是 48480 和 58848，均在失败 Adam 更新之前。旧交换评分使用主导质量相减的分母，可能得到有限但错误的极小正数，导致有限前向损失在反向时溢出。仅检查 NaN/Inf 无法覆盖这种情况。

现在先与正质量汇总的分母比较，检测到严重抵消时使用等价 log 混合重算代理收益；没有关闭梯度检查、跳样本或裁剪目标，原梯度 max_norm=1 保持。seed17 失败样本与独立 CPU 正质量参考最大差异 3.68e-26，完整梯度更新有限；seed29 从 58848 起回放 96 步通过，覆盖原失败事件。普通 City 单视频 12 步回放的模型、Adam、RNG、指标逐位一致。证明保存在 execution_optimization/gradient_failure/ 与 prefix_contract.json。

两路从安全 checkpoint 接续，失败之后未保存的更新按原顺序重放。历史 failure.json 和日志仍保留用于审计；当前运行状态查看 training/C_s17/status.json、training/C_s29/status.json 和 pipeline/status.json。下一阶段四卡分配保持不变。

## Full_Pset 四卡阶段已启动

入口为 `local_experiments.full_coverage.full_pset_pipeline`：seed17 在 GPU0–1、seed29 在 GPU2–3，各两进程 DDP、有效 batch=2，Adam 1e-4、clip=1，两个完整 epoch，每 seed 33,408 次优化器更新。冻结已导出的各自 C、Risk、配套 Adaptive φ 与 B16，以各自 E48 Full_Pset Reader 热启动，仅训练 51,157 参数 Reader，使用原 Set-only 目标。双卡新阶段不声称与旧单卡 Adam 逐位一致。最终 checkpoint 预先指定，训练完成后在 GPU4/5 各自跑完整开发集；GPU6/7 保留。

状态：`artifacts/full_coverage_v1/full_pset_pipeline/status.json` 与 `training/Full_Pset_s{17,29}/status.json`；运行协议：`full_pset_protocol.json`。训练有原子检查点和断点恢复，不读取 Test。
