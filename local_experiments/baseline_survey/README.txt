正式结果入口
  artifacts/ar_official_baselines_20261029/report.zh.md

一条命令（使用已有环境、权重、官方源码和固定输入；支持跳过已完成视频）
  cd /data/chenjiayu/zhikun/streaming_sparse
  CUDA_VISIBLE_DEVICES=1 bash local_setup/run_official_baselines.sh

指定新结果目录
  CUDA_VISIBLE_DEVICES=1 bash local_setup/run_official_baselines.sh artifacts/my_official_run

单方法调试例子（输出必须是新的目录）
  CUDA_VISIBLE_DEVICES=1 bash local_setup/run.sh local_experiments/run_official_baseline.py \
    --method dasa --chunks 8 \
    --inputs artifacts/ar_official_baselines_20261029/sailboat/shared_inputs.pt \
    --output artifacts/my_dasa_check

环境
  Python: streaming_sparse/.venv/bin/python
  共用权重: checkpoints/self_forcing_dmd.pt 的 generator_ema
  模型文件: wan_models/Wan2.1-T2V-1.3B
  源码: external_baselines/LightForcing, external_baselines/DASA, external_baselines/Forcing-KV
  DeepForcing 仅下载和调研，没有计入本次实测。

算法和预算
  原始完整 inference pipeline；不额外加入 rebase 组。
  dense / Light Forcing / DASA / Forcing-KV，共 3 prompts × 1 seed × 4 方法。
  持久 self-KV 上限 5,096,079,360 bytes；不含算子临时张量或文本 cross-KV。
  Forcing-KV 保留较小官方配置，实际 KV 低于上限。
  DASA 的受限缓存滚动错误已修复；补丁、未修改版本失败日志和前缀数值一致性审计均保留。
  所有 dense attention 用 PyTorch SDPA；LF 稀疏部分仍为官方 Triton kernel。
  同 Self-Forcing 权重；LF 论文微调权重、DASA 可选训练适配均未使用。

复现边界
  运行时记录每个方法源码、权重、输入哈希和配置。
  图表指标是描述性筛查，不是完整 VBench 或论文加速比复现。
  源码/补丁审计、环境依赖和 GPU 信息都在结果目录。
  共享 GPU 的耗时不应解释为独占硬件的性能排名。

文件分工
  run_official_suite.py: 完整实验顺序、固定输入复制、统一 VAE 解码。
  run_official_baseline.py: 独立进程导入官方模型与 pipeline、预算配置、审计 hooks。
  analyze_official_baselines.py: 视频指标、DINO、联系图、四宫格、结果图。
  audit_official_baselines.py: 视频帧数、输入/权重一致、预算和 DASA 回归校验。
  build_official_report.py: 从完成数据生成中文 Markdown 报告。
  sources/: 官方 README 和论文快照。

旧试跑
  artifacts/ar_sparse_baselines_20261029 带 SUPERSEDED.txt；其适配器视频不属于正式结果。
  该目录保留的 shared_inputs.pt 被正式 suite 按哈希复用，不复用旧视频。
