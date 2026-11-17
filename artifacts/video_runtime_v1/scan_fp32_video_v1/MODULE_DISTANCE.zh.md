# FP32候选扫描消费：部署验收与实现位置

本轮采用scan_fp32，Reader仍使用完整WORK；P2两种小WORK均无净收益，结束该方向。算法、权重、候选数、正质量扫描和FP64归约树不变。本改动没有引入新的近似。

## 改动与入口

- `local_experiments/video_runtime_v1/exclude_scan.py`：读取FP32候选字段后立即提升FP64，再作原乘法、前后缀扫描及主导页处理；实际值输出和fake值输出显式FP64，log输入/输出仍FP64。
- `reader_reduce.py::compact_scores`：加入侧直接传入CAND的FP32 strided视图，消除custom-op之前的FP64 ca中间张量；移出侧cr+pr仍先提升到FP64。编译器消除未使用的ca转换，完整路由验收使用编译路径。
- `deployment.py::CURRENT/configure`：scan_fp32默认开启；`benchmark_hotspots.py`提供显式`--scan-fp32`，通用实验默认关闭以便对照。
- `profile_scan_fp32.py`：同一捕获P432输入、tile72，预热各窗口后完整路由和profile；`run_scan_fp32_video.py`：GPU6顺序运行基线与候选，单短视频、预热后两次测量。
- `quality_scan_fp32_video.py`：相对本轮基线及此前验收基线核对latent/RGB并计算指标。

## 实测收益与理论口径

|项目|改前|改后|加速比/变化|
|---|---:|---:|---:|
|完整路由CUDA Graph中位数|4.313888 ms|4.246976 ms|1.0158×，省0.066912 ms|
|排除扫描kernel合计（独立profile）|0.243573 ms|0.245110 ms|0.9937×|
|FP32→FP64转换kernel（profile）|0.072572 ms|消除|直接省去转换阶段|
|9 chunks生成|10.550877 s|10.493210 s|1.0055×|
|VAE|4.316247 s|4.306765 s|1.0022×|
|生成＋VAE|14.867124 s|14.799975 s|1.0045×|

完整路由50次CUDA Graph中位数。profile和计时独立，不能逐微秒严格相加。视频为GPU6同卡、9chunks/105帧/768×512/16FPS、history360、tile72；VAE和生成分别预热取中位数后相加，不含编码和文件IO。VAE未改，其约9.5ms变化是测量波动。没有测15s/30s。

ca有864×16×32×16=7,077,888个元素。移除转换的逻辑读取28.312MB＋写出56.623MB=84.935MB，按已有1266.396GB/s复制参考约0.0671ms。本轮完整路由省0.0669ms，与这笔工作量同量级；这是流量参考，不是独立测得的HBM流量或普适下限。扫描本身没有加速，主要收益来自不再物化转换张量。

Reader仍约1.9ms，不能用WORK单项流量下限推断可回收毫秒数。P2完整路由4.3377→4.4980/4.3752ms，无收益；见[有界实验报告](../p2_bounded_v1/REPORT.zh.md)。本轮不再扩大优化搜索。

## 数值验收

完整路由5个输出逐位一致。独立扫描覆盖K16/K32、普通和主导页位于头/中/尾、1000 log质量差距、FP32非连续字段；与先转double输入的基线逐位一致，实际与compiled fake契约输出FP64。见[contracts.json](../scan_fp32_v1/contracts.json)。

短视频基线与上一轮已验收latent一致；优化前后latent、解码RGB均逐位一致。latent NMSE=0，RGB MAE/max=0，LPIPS=0，SSIM=0.999999572（指标实现舍入），PSNR=100dB（实现上限，逐位一致理论为∞）。见[errors.json](errors.json)、[quality.json](quality.json)。这里评估优化误差，不是算法相对Dense误差。

## 部署和质量清单

batch.py已从deployment.CURRENT继承tile72/history360，透传reader-checkpoint、history-pages、local-assets，沿用现有identity；README改为9chunks示例与全覆盖权重说明。实际84任务验证入口，不额外重跑P0性能测试。

[84视频质量报告](../quality_batch_9d4d33d_v1/REPORT.zh.md)冻结9d4d33d源代码，仅叠加batch入口修正，不包含本轮scan和P2。当前scan通过独立配对短视频验收后部署，不能把84视频称为scan质量测试。

## 原始记录和复现

- 路由/profile：`artifacts/video_runtime_v1/scan_fp32_v1/scan_p432_steady_detail`。
- 视频生成、独立插桩、decode：本目录`baseline/`和`scanned/`。
- 当前部署命令：`local_setup/CURRENT_VIDEO_BASELINE.txt`，常驻worker通过共享deployment调用。
- 脚本入口：`python -m local_experiments.video_runtime_v1.profile_scan_fp32`、`python -m local_experiments.video_runtime_v1.run_scan_fp32_video`、`python -m local_experiments.video_runtime_v1.quality_scan_fp32_video`；使用项目venv，并先核对脚本输出目录，避免覆盖本次证据。

编译消费边界证据：[生成代码与说明](../scan_fp32_v1/compiler_evidence/README.md)，确认7,077,888个FP64 ca元素不再物化。
