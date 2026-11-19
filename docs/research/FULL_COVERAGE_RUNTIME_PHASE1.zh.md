# 全覆盖扩展与第一轮运行时优化

## 当前顺序

按用户后续指引，全覆盖 C 训练独立继续，运行时优化先用一份兼容 C pooling 权重广播占位，不等待全覆盖成品。占位结果只能验证工程路径及耗时，不能声称是正式全覆盖模型质量。正式扩展范围仍为 29 个稀疏层 × 12 个 head（348 个位置），Layer 0 继续 Dense。

入口：`local_experiments/full_coverage/`。旧 E24–E49 的代码和结果不被覆盖，新产物全部写入 `artifacts/full_coverage_v1/`。

## 固定的数据与训练范围

沿用 E24 的 prompt、noise seed、chunk 2/5/8、step 0/2 和每个状态的两个 Query tile，扩展 layer=1..29、head=0..11。8 条训练轨迹、2 条开发轨迹，分别 33,408 / 8,352 个事件。每条轨迹包含 4,176 个事件，每个位置 12 个事件。Test 不采集、不评价。

数据由每张 GPU 一个常驻 `CaptureRuntime` 采集；模型在该 worker 内复用。采集使用现有 CachedLatestRuntime 的冻结 summary_pretrained/G-KVR 基线，历史 338 页、当前 72 页；不是重放旧 E24 文件逐位相同轨迹的声明。当前 KV scratch、page64 精算与完整均值补偿均复用仓库实现。FP64 teacher 分 head/tile 计算，保存逐页 logZ/output 和 page_log_center_v3 参考量。

C 初始化从每个位置自己的原 pooling 权重开始；共享 E24 Reader 从对应 seed 的 A 预训练权重初始化。仅训练 pool_k、pool_v、pool_feature 和共享 Reader，冻结 grouping、bank、basis、theta 和视频 backbone。沿用 E24 C 联合损失与 Adam 配置；训练两次完整 epoch，每个位置实际参与 192 个事件更新。新增数据后总更新数为 66,816，不再是原子集的 2,304；不能声称两者总训练预算相同。B16 和尺度继续复用原 E24 train-only fit，不用开发集重拟合。

训练每 96 步保存模型、Adam、随机状态、梯度覆盖和曲线；失败停止并保留日志。重新执行从完整轨迹和 latest.pt 恢复。每个位置的三个 pooling 参数都必须至少出现一次有限非零梯度。

导出 `deployment.pt` 前检查 348 个位置的完整训练覆盖、每个位置开发集 24 个事件、非训练参数逐位冻结，并要求所选 epoch 的开发集 C+Risk NMSE / 原基线 NMSE <= 1.02。checkpoint 按该初选质量选择，导出为原生 Model state_dict，无未训练位置回退。该产物只表示 C+Risk；旧 E48 Full_Pset 尚未针对新摘要和全覆盖分布适配，不能直接标记为全覆盖 Full_Pset。

## 运行与检查

使用现有实验环境：

```bash
PYTHON=/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
$PYTHON -m local_experiments.full_coverage.pipeline --gpus 0 1 2 3 4 5 6 7
```

管线依次执行全覆盖采集、数据审计、seed17/29 C 联合训练和严格部署导出。可检查：

- `artifacts/full_coverage_v1/pipeline/status.json`
- `artifacts/full_coverage_v1/workers/capture*/status.json` 与 `run.log`
- `artifacts/full_coverage_v1/trajectories/*/progress.json` 与 `complete.json`
- `artifacts/full_coverage_v1/training/C_s*/status.json`、`complete.json` 与 `export.json`

预检已验证全部 348 个位置的初始化与 trainable 参数边界；完整联合损失预检覆盖 l1h0、原来未覆盖的 l2h1/l23h11，512 个动作、own 保护、三个 pooling 梯度、真实 optimizer 更新及其余参数冻结均通过。预检更新全部丢弃，不进入正式模型。原始记录在 `preflight.json`、`joint_preflight.json`。

## 后续仍需完成

全覆盖 C 成品、开发质量与导出仍以管线实际 complete/export 文件为准，随后需要对冻结新 C 适配 Reader/Full_Pset。缓存页面系数、共享 Q64 logits、共享背景评分、块稀疏精算、完整均值补偿和常驻 worker 已在 `local_experiments/video_runtime_v1/` 实现。优化记录见 `docs/research/evidence/video_runtime_v1/REPORT.zh.md`。以后加速测试只使用单个视频，可只隔离一个模块计时，不扩展多场景测速矩阵。

速度报告需要在相同输入、相同权重、相同有效页数和同一空闲 GPU 上比较参考与优化路径，区分编译预热和稳态，并记录摘要/编码、路由/修正、Attention、clean commit、VAE 解码及整视频耗时。采集和训练耗时不是运行时加速数据。本轮已有单视频原版/优化版耗时与模块记录；正式全覆盖视频质量仍未验证，且占位权重下原型与优化版完整闭环 latent 存在差异，不能据此声称质量等价。
