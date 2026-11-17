# 缓存 R4 log质量：本轮验收与模块性能距离

## 配置与测量口径

当前基线保留用户指定旧快精算/补偿：Page64/Q64/R4、完整128维V均值＋rank8、M16/B16、10%精算、16×32交换＋no-op；全覆盖C_s17＋Full_Pset_s17。旧RoPE、A-only/split Reader、scan-exclude、projected-risk开启；PAGE32、lazy关闭。本轮只增加 --cached-logmass，权重和候选规则均未改变。

模块输入为真实轨迹捕获的12 heads ×72 Q64 tiles ×M16 ×432页（360历史页＋72当前页）。路由数据来自本轮同一输入完整profile；Attention使用相同未变实现此前隔离测量。每个模块只计一次，custom-op外层与内部kernel不重复加；Reader子阶段不再叠加到合计。完整路由CUDA Graph交替预热、各50次稳态计时。

设备A100-SXM4-40GB。采用本机实测参考：复制1266.396 GB/s、BF16大GEMM 202.135 TFLOP/s、FP32 13.4865 TFLOP/s、FP64 13.6845 TFLOP/s，来源 ../lazy_video_v1/device_roof.json。以下下限由主要FLOPs或最低明确逻辑读写量推导；未包含的exp/log、归约通信、访问布局会降低实际可达速度。距离不是可承诺加速比，差额也不是实际可节省时间。逻辑payload不等于HBM计数器流量。

## 模块耗时、理论距离与优化空间

|模块|实测/ms|参考下限/ms|距离|参考差额/ms|优化空间与优先级|
|---|---:|---:|---:|---:|---|
|旧快精算＋完整均值/rank8补偿|1.295|0.771|1.68×|0.524|中低：已换快实现，先处理Reader|
|Reader标量＋向量|1.938|0.447|4.34×|1.491|高：系数复用与归约组织；向量1.448ms、标量0.490ms|
|Reader log质量（本轮两阶段）|0.223|0.189|1.18×|0.034|低：已消除R4重复计算；继续优化受非线性及归约限制|
|Risk主要路径，含概率/查询预处理和U投影|0.470|0.377|1.25×|0.093|低：保留FP32 SGEMM，主路径接近逻辑payload参考|
|512动作标量评分|0.337|0.089|3.77×|0.247|中：exp和中间量读写；前轮融合已筛查，收益很小|
|动作log分母生成|0.324|0.045|7.26×|0.280|中：距离大但下限只算输出，不能推断7倍可达收益|
|排除候选正质量扫描|0.292|0.143|2.05×|0.149|中低：减少重复访问/扫描通信，保留主导页稳定处理|

其他路由开销：probe QK约0.155ms、FP64交叉矩BMM约0.159ms、矩/增量组合约0.210ms、候选池topk/sort等约0.2ms；部分操作在融合kernel内。它们未做完整逐算子工作量审计，故不填虚构理论倍数。动作回退0.033ms，不是主要热点。表内行不是完整路由的无遗漏分解。

### 下限计算依据与历史修正

1. Attention：主要BF16发射工作量155.855 GFLOPs /202.135 TFLOP/s =0.771ms。包含当前旧快精算和补偿，两者合计实测1.295ms；不是早期较慢stream_attention的239.176 GFLOPs。
2. Reader两阶段：WORK约283.116MB，写入＋读取566.231MB /1266.396GB/s =0.447ms。未包含系数和exp/tanh，4.34倍只是部分工作量距离。向量阶段仍是最值得深入的模块；此前PAGE32、向量宽度/寄存器筛查结果应保留，避免重复无效实验。
3. 本轮log质量：输入Z为12×72×16×432×4×4字节=95.551MB，FP64局部质量47.776MB。局部写、归一化读、最终写各47.776MB，总238.879MB，对应0.189ms。profile局部生成0.146ms＋归一化0.077ms=0.223ms，距离1.18倍。若只按唯一输入＋最终输出143.327MB算单核参考，则0.113ms、距离1.97倍；这忽略了本轮实际使用的中间缓存。单核融合已试，未比两阶段更快。
4. Risk：概率相关Z读/prob写/prob读286.654MB＋保留U投影的写/读191.103MB =477.757MB，对应0.377ms。主要kernel合计0.470ms。历史表把与Reader共享的log质量计算归进纯Risk，且漏计U，不能沿用原3.86倍结论。
5. 动作评分：主要den/dot输入113.246MB，下限0.089ms；log分母输出56.623MB，下限0.045ms，未计输入及多次logaddexp。编译器已经共享第一步质量组合；前轮20组融合筛查无明确整体收益，因此不以表中大距离直接判断值得重写。
6. 扫描：主要唯一输入/输出180.486MB，下限0.143ms；GPU并行扫描存在logK通信步骤。历史排除BMM的0.566GFLOPs对应0.041ms仅为算术下限；计入311.427MB逻辑payload后约0.246ms，原0.372ms实测距离约1.51倍，而不是可兑现9倍。扫描替代是改变组织方式带来的收益。

## 本轮针对性优化及隔离验证

原部署Reader log质量kernel三遍扫描页面，每遍重算R4局部logsumexp。本轮用独立FP64 kernel只计算一次局部log质量，后续页面归一化继续由原torch表达式编译。没有低精度近似；libdevice exp/log和稳定最大值缩放保留。两阶段/单阶段共8组布局筛查，选择两阶段BLOCK256、4warps；2warps的配对倍数略高但其绝对时间更慢且不同GPU，未据此替换。

|页数|独立原表达式/ms|两阶段/ms|加速|最大绝对差|
|---:|---:|---:|---:|---:|
|216|0.136|0.123|1.108×|1.42e-14|
|288|0.218|0.173|1.260×|2.27e-13|
|360|0.220|0.199|1.108×|1.42e-14|
|432|0.249|0.229|1.085×|2.27e-13|

完整路由 4.915→4.580ms，1.073×；捕获事件5项路由输出全部逐位一致。独立表达式编译和部署编译生成不同归约策略，所以独立模块约0.249→0.229ms不能直接替代部署profile约0.560→0.223ms。完整路由的0.335ms收益与部署profile相符。

## 短视频验收

GPU3同卡顺序运行，city_walk_s701，9chunks/105帧，768×512、16FPS，5个历史chunks/360历史页；只跑短视频。完整预热后两次生成计时，VAE独立预热测量，合计不含模型加载、编译、文件读写、视频编码。阶段累计来自独立插桩运行，不与墙钟简单求和。

|阶段|本轮前/s|本轮后/s|加速比|
|---|---:|---:|---:|
|生成|10.937|10.735|1.019×|
|VAE|4.308|4.308|1.000×|
|生成＋VAE|15.245|15.043|1.013×|
|route_and_correction|3.676|3.463|1.062×|
|attention|1.851|1.851|1.000×|
|summary_and_encoding|0.442|0.447|0.990×|
|clean_commit|0.875|0.876|0.998×|

摘要/编码、clean commit、VAE包含多个不同网络/数据移动阶段，尚无完整工作量普查，不给统一理论下限。它们本轮未改动，小幅波动不归为优化收益。

|优化误差指标|结果|
|---|---:|
|latent NMSE|0.0|
|RGB MAE /255|0.0|
|RGB最大绝对差 /255|0.0|
|PSNR/dB|100.0（零差封顶）|
|SSIM|0.999999572|
|LPIPS|0.0|

latent和全部RGB帧逐位一致；SSIM接近1的小偏差来自指标实现。LPIPS/SSIM/PSNR沿用[21,101)评价窗口。基线latent与上一轮已接受projected-risk输出逐位一致。隔离FP64 log质量约1e-14–2e-13差异来自计算/归约次序，未降低精度；本次完整短视频无新增误差。这里比较优化前后，不讨论算法相对Dense的近似误差。

## 采用结论、复现及暂停点

本轮采用 --cached-logmass，通用实验开关默认仍关闭，当前完整命令更新到 local_setup/CURRENT_VIDEO_BASELINE.txt。精算和补偿继续使用旧快实现。下一优先级仍是Reader向量系数复用，其次动作分母/评分；本轮结束后按用户要求暂停目标，不启动这些工作。

证据：acceptance.json / errors.json / quality.json；../logmass_cached_v1/各配置result.json、full_route/before_profile.txt、after_profile.txt、result.json；原编译重复计算证据在../risk_projected_video_v1/compiler_evidence/。源码快照与哈希在source_snapshot/、source_hashes.json。脚本为check_logmass_cached.py、profile_logmass_route.py、run_logmass_video.py、quality_logmass_video.py、report_logmass_video.py。

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

