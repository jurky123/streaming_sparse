# 2026-11-02 源码与报告发布

本次上传 `local_experiments/` 的研究/训练/推理/kernel与评测源码、`local_setup/` 环境入口、固定外部方法版本和兼容补丁，以及Markdown报告和精简JSON证据。未上传模型权重、捕获张量、原始输入、视频、日志、运行队列、环境目录或下载的论文。

## 当前接受的实现

`local_experiments.gap_gain_vector64_runtime.GapGainVector64Runtime`。
90%精确页稀疏率、16 probes、每页4个代表、低秩未选中贡献补偿；所有稀疏模块使用已有训练权重，骨干冻结。
本次没有新的恢复训练。`GapReorderRuntime` 只是实验候选，完整计时没有收益，未纳入当前版本。

- [最新方法比较](streaming_evaluation_20261102/report.zh.md)
- [向量融合验收](optimization_20261102/vector64.zh.md)
- [探针投影验收](optimization_20261102/route_qp.zh.md)
- [重排负结果](optimization_20261102/reorder_rejected.zh.md)
- [模块profile成本](optimization_20261102/cost_accounting.zh.md)

本轮主要硬件为NVIDIA H200 NVL，和2026-10-26历史发布文档中的A100环境不同；不同日期/环境结果不能直接拼成加速比。

## 关键入口

|入口|用途|
|---|---|
|`local_setup/run.sh`|Python/CUDA环境入口，建议显式指定`CUDA_VISIBLE_DEVICES`|
|`local_setup/fetch_stream_baselines.py`|获取固定版本官方基线并应用归档补丁|
|`local_experiments/stream_eval_official.py`|官方完整管线、共享输入、原生VAE、两轮完整预热后计时|
|`local_experiments/stream_eval_ours.py`|当前稀疏实现使用相同输入的对应入口|
|`local_experiments/gap_stream_eval_suite.py`|串行隔离子进程的三场景可恢复评测|
|`local_experiments/gap_stream_eval_quality.py`|前段独立描述指标和对比视频|
|`local_experiments/stream_eval_report.py`|从本地实验输出生成Markdown和JSON汇总|
|`local_experiments/gap_gain_vector64_perf.py`|当前接受版本的配对端到端验收|

`gap_gpu4_session.py` 是本机显存预留/串行队列工具，可通过环境变量指定设备和状态目录；名称中的GPU4不代表只能用GPU4。部署时不要直接复制本机PID、活动队列或默认显存预留策略。

## 复现实验所需资产

新克隆只有源码和报告，**不能直接复现已训练模型的结果**。仍需恢复或重新生成：

- `wan_models/Wan2.1-T2V-1.3B/` 与 `checkpoints/self_forcing_dmd.pt`，公共权重来源见现有migration下载工具。
- `artifacts/full_route_grad_20261031/epoch1.pt` 初始化和 `artifacts/four_gaps_20261031/shared/epoch1.pt` 稀疏模型。
- 训练/字典等源码引用的先前研究资产；目录命名记录原始实验依赖，不保证仅两个checkpoint就足够。
- 三场景 `artifacts/ar_official_baselines_20261029/<场景>/shared_inputs.pt`（噪声和嵌入）。本次直接复用它们，没有重复上传。
- LPIPS与DINO评估权重。DINO路径在评测代码中保留本机配置，需要迁移时调整。

骨干、输入及稀疏checkpoint哈希、官方固定commit和方法限制保存在 `docs/streaming_evaluation_20261102/`。缺失资产时不要绕过校验，也不要把随机初始化结果当成本文报告的模型。

完整端到端含生成和统一VAE，不含加载/编译、视频编码或质量打分。前段主质量窗口为RGB[21,101)。方法比较只有三个场景和两轮计时；原生缓存/位置/稀疏策略有明确差异，不能冒充所有方法实际保留相同token的论文复现。

## 本次验证范围

提交前检查全部新增Python语法、Shell语法、报告相对链接与上传内容；实际GPU验证见已归档报告（15个场景-方法单元、30轮计时）。没有为了上传代码重新执行GPU训练或完整评测。
