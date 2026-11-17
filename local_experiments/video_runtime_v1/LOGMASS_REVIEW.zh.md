## 评审代码导航与本轮改动边界

以下路径从本报告所在的 `artifacts/video_runtime_v1/logmass_video_v1/` 解析。提交包括此前尚未提交的运行时，因此不能把整个提交的代码量当成本轮新增工作；**本轮性能改动只是在路由中缓存一次 R4 局部 log质量**。

|评审对象|源码与函数|职责与本轮状态|
|---|---|---|
|本轮新 kernel|[logmass_cached.py](../../../local_experiments/video_runtime_v1/logmass_cached.py)，`local_mass`、`local`、`normalized`|`local_mass`读FP32 Z、转FP64，稳定计算4个代表的logsumexp；`local`分配FP64临时结果并发射kernel；`normalized`做页面维归一化。采用BLOCK256、4warps、禁用FP融合。|
|未采用对照|同文件 `fused_mass`、`fused`|单kernel计算局部质量及页面归一化；`FUSED=False`，8组筛查中没有胜过两阶段方案。|
|路由接入|[pipeline_graph.py](../../../local_experiments/video_runtime_v1/pipeline_graph.py)，`BatchGraphs.compute`|`CACHED_LOGMASS`分支调用`normalized(z)`；关闭时保留原`zz.logsumexp(-1)`及页面归一化。结果lm仍传给同一个`compact_scores`。|
|配置入口|[runtime.py](../../../local_experiments/video_runtime_v1/runtime.py)，`VideoRuntime.__init__`；[benchmark_hotspots.py](../../../local_experiments/video_runtime_v1/benchmark_hotspots.py)|`cached_logmass`默认为false，CLI `--cached-logmass`显式启用，并写入结果元数据。运行时设置模块级开关，实验按每进程单视频运行，不用于同进程并发混合配置。|
|页面/KV缓存|[page.py](../../../local_experiments/video_runtime_v1/page.py)、[runtime.py](../../../local_experiments/video_runtime_v1/runtime.py)|已有clean commit页面摘要/编码和驻留KV生命周期；本轮没有改动。这种历史缓存不同于本轮每次forward临时缓存的query-dependent局部log质量。|
|旧快精算/补偿入口|[legacy_components.py](../../../local_experiments/video_runtime_v1/legacy_components.py)，`attention`|`--legacy-components rope,attention`选择该路径，直接读驻留KV和页LUT。沿用已接受基线，本轮没有换Attention算法或精度。|
|精算kernel|[gap_exact_round_kernel.py](../../../local_experiments/gap_exact_round_kernel.py)，`_attn_fwd`|Page64/Q64，入口设置`HIGH_PRECISION_AV=True`，AV使用概率BF16高低项两项乘积；输出精算结果与base2 LSE。|
|完整均值补偿与合并|[tuned_attention.py](../../../local_experiments/review_v1/tuned_attention.py)，`_tail`、`_merge`|先加权完整128维V均值和rank8细节，再重建细节并按LSE合并；不是用rank8替代完整均值。沿用已有BF16近似路径；本轮新增log质量则全程FP64。|
|Risk|[risk_projected.py](../../../local_experiments/video_runtime_v1/risk_projected.py)，`risk`、`reduce`、`kernel`|原FP32 SGEMM投影＋融合特征/M16归约，本轮不变。|
|Reader|[reader_reduce.py](../../../local_experiments/video_runtime_v1/reader_reduce.py)，`compact_scores`、`reader_reduce`；[reader_split.py](../../../local_experiments/video_runtime_v1/reader_split.py)，`scalar_kernel`、`vector_kernel`|A-only布局、两阶段Reader；消费新旧实现同形状的FP64 lm，本轮计算公式不变。|
|候选、交换评分|[scores.py](../../../local_experiments/video_runtime_v1/scores.py)，`candidate_pool`、`apply_best`；[scores_moments.py](../../../local_experiments/video_runtime_v1/scores_moments.py)，`moments`|16移出×32加入，最高正收益交换，否则no-op；本轮不变。|
|稳定排除聚合|[exclude_scan.py](../../../local_experiments/video_runtime_v1/exclude_scan.py)，`exclude_scan`、`scan_kernel`|正质量前后缀扫描、主导页移出时单独缩放；本轮不变。|

### 数学定义与输入输出约定

输入 `z[h,t,m,p,r]` 为连续FP32，形状 `[12,72,16,P,4]`。设

```text
v[h,t,m,p] = logsumexp_r(float64(z[h,t,m,p,r]))
lm[h,t,m,p] = v[h,t,m,p] - logsumexp_p(v[h,t,m,p])
```

输出 `lm` 为FP64、形状 `[12,72,16,P]`。本轮不改上述表达式，只让 `v` 写入一次临时张量，阻止编译器在页面归一化的多遍扫描中重复计算它。临时张量由CUDA Graph捕获后复用；不是跨Query、step或视频复用响应。历史clean页的静态编码仍由原缓存维护。

`local_mass` 用 `m=max(z0,z1,z2,z3)` 和 `m+log(sum(exp(zi-m)))` 保持稳定；exp/log使用FP64 libdevice。代表质量16的共同因子在页面归一化中相消，因此这里不需要额外乘16；Attention补偿自己的质量因子保持原实现。本轮隔离差异是求和/编译顺序引起的FP64舍入，不是降精度近似。完整视频验收只证明已测短视频逐位一致，不声称所有输入都逐位等价。

### 如何复查证据和复现

|检查项|脚本|结果文件与判据|
|---|---|---|
|局部公式和四窗口速度|[check_logmass_cached.py](../../../local_experiments/video_runtime_v1/check_logmass_cached.py)|[隔离结果目录](../logmass_cached_v1/)；各`result.json`包含输入shape、误差、50次原始计时。|
|完整路由收益与输出|[profile_logmass_route.py](../../../local_experiments/video_runtime_v1/profile_logmass_route.py)|[路由result.json](../logmass_cached_v1/full_route/result.json)及前后profile；`output_equal`检查`BatchGraphs.compute`五项返回值。|
|同卡短视频前后对照|[run_logmass_video.py](../../../local_experiments/video_runtime_v1/run_logmass_video.py)|[baseline/result.json](baseline/result.json)、[cached/result.json](cached/result.json)，两组差别仅`--cached-logmass`。|
|感知指标|[quality_logmass_video.py](../../../local_experiments/video_runtime_v1/quality_logmass_video.py)|[quality.json](quality.json)：LPIPS/SSIM/PSNR窗口[21,101)；[errors.json](errors.json)：全latent和全部RGB帧逐位比较。|
|报告重建|[report_logmass_video.py](../../../local_experiments/video_runtime_v1/report_logmass_video.py)|读取已保存JSON，生成本报告与[acceptance.json](acceptance.json)，不重新跑GPU。|

完整基线命令见 [CURRENT_VIDEO_BASELINE.txt](../../../local_setup/CURRENT_VIDEO_BASELINE.txt)。仓库根目录运行：

```bash
# 局部隔离；其他窗口用 --pages 216 / 288 / 360
CUDA_VISIBLE_DEVICES=0 python -m local_experiments.video_runtime_v1.check_logmass_cached --pages 432 --block 256 --warps 4
# 原始路由profile；使用已保存实际轨迹输入
CUDA_VISIBLE_DEVICES=3 python -m local_experiments.video_runtime_v1.profile_logmass_route
# 单卡顺序跑两组短视频；先给脚本OUT换一个新目录以保留本次证据
CUDA_VISIBLE_DEVICES=3 python -m local_experiments.video_runtime_v1.run_logmass_video
# 只重建报告
python -m local_experiments.video_runtime_v1.report_logmass_video
```

复现GPU测试需要本机权重、`route_entry.pt`、Wan模型和CUDA/PyTorch/Triton环境。推送包含源码、配置和紧凑文本证据，不包含模型权重、latent、像素数组和视频；这些大文件仍在实验机原路径。权重摘要可查两组result.json的`weights`字段。`source_hashes.json`是本轮验收/报告生成源码快照，实际测试时benchmark保存的`source`字段提供运行时源码证据；报告文字更新不改变原测量值。
