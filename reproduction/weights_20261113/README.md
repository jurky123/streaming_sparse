# 已训练权重与实验复现

本包对应代码基线 `04e710f`。每个 `.pt.gz` 是原 checkpoint 的无损压缩，manifest.json 保存压缩前后 SHA256 和恢复路径。直接包含于 Git，无需 Git LFS 或私人存储服务。所有原始精度、state_dict 和已有 optimizer 状态均保留。

## 配置对应关系

|资产名|用途|
|---|---|
|summary_pretrained|**用户选定配置**：摘要预训练，无后续下游训练|
|full_calibrated|两步 x₀ 蒸馏后、校准过的完整方案；用于已报告的 full 消融基线|
|two_step_distilled|两步 x₀ 蒸馏后的未校准 checkpoint|
|downstream_router_pilot|仅6层×3 heads下游排序监督的试验，已有视频结果退化|
|single_step_flow|单步 flow 蒸馏试验，已有视频结果退化|
|legacy_sparse|新结构初始化、旧版对照与低秩均值消融所需参数|
|legacy_route_initializer / legacy_compensation / legacy_dictionary / legacy_pool|旧版 runtime 初始化依赖|
|embedding_*|四个训练/验收场景的原始文本 embedding，避免不同 T5 环境引入差异|

选定配置保留硬重排分组、4个页内代表、完整V均值、3×8细节编码及原风险路由。512×768，历史338页（layer0为360），每页64 tokens，16 probes，10%精确页预算；前两chunk及clean commit使用dense。骨干冻结。实际持久化KV/摘要预算由运行结果检查。

## 新环境安装和恢复

Linux + Python 3.10，CUDA GPU。原验收为H200、PyTorch 2.6.0+cu124；依赖精确版本见 environment.json 和 migration/requirements.lock.txt。运行脚本要求磁盘剩余至少50 GiB，建议为公开基础模型额外准备足够空间。GPU显存需求取决于加载峰值，尚未验证较小显存卡；不要把持久化KV预算当成整模型显存需求。

```bash
git clone https://github.com/jurky123/streaming_sparse.git
cd streaming_sparse
bash reproduction/setup.sh
# 固定revision下载Wan、Self-Forcing骨干、T5、VAE/tokenizer，检查SHA256
bash local_setup/run.sh local_setup/download_models.py
```

`setup.sh`不运行旧版 migration/verify.py 的陈旧源码清单检查。公开模型仍需可访问 Hugging Face；LPIPS 首次调用会从 PyTorch 下载 AlexNet，并使用 lpips==0.1.4 自带线性校准权重。CPU环境可只运行 `python3 reproduction/restore_weights.py` 检查和解压训练资产。

## 重现选定配置及消融

```bash
# 指定你有权使用的GPU；先生成同输入dense，再生成选定配置
CUDA_VISIBLE_DEVICES=0 bash local_setup/run.sh reproduction/run_quality.py \
  --case horse_field --variant summary_pretrained
CUDA_VISIBLE_DEVICES=0 bash local_setup/run.sh reproduction/run_quality.py \
  --case traffic_crossing --variant summary_pretrained
# 可选：full_calibrated / mass_route / no_grouping / no_detail /
# projected_mean / no_new_training / downstream_router_pilot / single_step_flow
# --chunks 41 为长视频；默认9 chunks、seed101，评估RGB[21,101)
```

选定配置输出位于 `artifacts/review_v1_20261102/ablation/video/{case}_s101/no_downstream/`，dense 位于 `artifacts/review_v1_20261102/video/{case}_s101/dense/`；视频、逐帧和聚合 LPIPS/SSIM/PSNR 均由原始评估代码计算。短视频默认105帧，主指标窗口为[21,101)。已完成chunk支持重放恢复。已有不匹配的实验identity会拒绝覆盖，请使用干净clone或归档旧实验目录。

选定配置参考值（seed101）：

|场景|LPIPS↓|SSIM↑|PSNR↑ dB|
|---|---:|---:|---:|
|horse_field|0.250368|0.608774|16.097|
|traffic_crossing|0.336512|0.383358|15.527|

完整参考指标和对应代码/权重SHA见 docs/results_snapshot_20261113。不同GPU、CUDA和算子版本可能改变结果；本包不承诺跨硬件逐bit一致。该配置仍是FP32质量参考路径，已有共享GPU耗时不能用作端到端加速结论。

## 训练复现实验的边界

本包直接支持从已发布权重重跑推理、视频对照和去模块消融；不是全部历史训练作业的恢复目录。原权重中的optimizer状态保留，但逐事件checkpoint、所有早期初始化和中间标签并未全部打包，不能仅凭latest.pt声称可以从任意历史事件续训。

例如从选定摘要权重重新进行单步flow训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash local_setup/run.sh -m local_experiments.gap_flow_single_train \
  --case city_walk --initial artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt
CUDA_VISIBLE_DEVICES=0 bash local_setup/run.sh -m local_experiments.gap_flow_single_train \
  --case reef_fish --initial artifacts/flow_single_20261104/train/city_walk/model.pt
```

这会重新采集学生轨迹并生成逐事件checkpoint。原始权重文件是可信仓库发布的PyTorch pickle；勿将恢复脚本用于不明来源的替代文件。
