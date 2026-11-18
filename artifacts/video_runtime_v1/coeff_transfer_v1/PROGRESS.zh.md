# 补偿系数的线程布局

P432 满窗口在 64 个代表概率上，按每页 4 个代表归约得到均值系数和 3 个 detail 系数。四组系数各自拆为 BF16 高低项，经共享内存送入 AV。原布局中，每个线程持有 4 行、1 页的 4 个代表，warp 按 2 行 × 16 列分工。新布局让每个线程持有 1 行的 2 个相邻页，每页 4 个代表，warp 改为 8 行 × 4 列。相邻系数可成对写共享内存。四代表归约仍由同一线程完成，精算与双项 AV 算术顺序相同。[布局与部署实现](../../../local_experiments/video_runtime_v1/page_prefetch.py)

## 模块验收

A100，Q64、BR64、4 warps，基线启用上一轮 page_prefetch。P432 使用真实捕获输入，较小窗口从同一输入截取并将页号对窗口页数取模。每个候选同进程 CUDA Graph 交替运行，预热 300 次，20 对、每次 8 次重放。各版时间为中位数，加速比取配对比值中位数。[测试程序](measure.py)

| 页数 | 上版 ms | 新版 ms | 加速 | spill 槽，上版 → 新版 | 输出 |
|---:|---:|---:|---:|---:|---|
| 216 | 0.61427 | 0.59782 | 1.02795× | 88 → 70 | 逐位一致 |
| 288 | 0.76614 | 0.74483 | 1.02870× | 80 → 66 | 逐位一致 |
| 360 | 0.98810 | 0.95053 | 1.03952× | 88 → 74 | 逐位一致 |
| 432 | 1.12602 | 1.09594 | 1.02751× | 80 → 66 | 逐位一致 |

P432 每线程仍用 255 个寄存器，每 CTA 仍用 80 KiB 共享内存。仅成对持有系数、保留原 2×16 线程分工为 0.9769×。转置共享布局为 0.9109×，关闭 swizzle 为 0.9904×，上述输出逐位一致；直接从归约布局转换到 dot 操作数的三个变体未通过 Triton 3.2 的指令降低。极端第二页输入触发 anchor 回退，两版输出有限且相等；单 CTA 的 Racecheck 报告 0 hazards。[模块原始结果](aligned_432.json) · [回退测试](stress.json) · [Racecheck](racecheck.txt)

## 硬件计数

| P432 完整融合调用 | 上版 | 新版 |
|---|---:|---:|
| 补偿高低项构造行的 excessive shared wavefront | 5,971,968 | 0 |
| shared store bank conflicts | 12,607,953 | 3,944,955 |
| shared load bank conflicts | 1,050,011 | 4,012,148 |
| local load sectors | 1,105,920 | 898,560 |
| Tensor pipe active | 45.84% | 47.00% |

源码行计数取 Nsight Compute SourceCounters；全核计数来自相同设备的两次完整网格采样。目标行的额外 wavefront 消失，部分冲突转移到读取端，完整模块仍有净收益。[原版源码行](../fused_fragments_v1/pc_samples.json) · [新版源码行](aligned_source.ncu-rep) · [原版全核计数](control_ncu.csv) · [新版全核计数](aligned_ncu.csv)

按四窗口在视频内的调用次数加权，模块预测 797.22 → 774.52 ms；同卡各采一次完整 105 帧设备时间线，融合核累计 797.22 → 774.41 ms，收回 22.81 ms。相对上版 268.89 ms 的差额，本轮收回约 8.5%。新吞吐约 142.0 TFLOP/s，距 528.33 ms 的 Dense 有效吞吐折算仍差 246.08 ms。新调度中 28,824 个 short scoreboard 源码行样本里有 15,777 个仍落在精算双项 AV。[旧时间线](timeline_off.json) · [新时间线](timeline_on.json) · [新版指令采样汇总](aligned_pc_samples.json)

## 视频验收与默认入口

GPU7 的 city_walk / seed 701 按新旧交替顺序做 8 对 105 帧生成。8 对均更快，配对中位节省 22.35 ms（1.00445×），范围 8.35–28.41 ms。所有配对的最终 latent、每个 chunk 与运行计数均一致，本轮采用单视频验收。新旧 latent 对同主干 Dense 的 NMSE 均为 0.378409，MAE 均为 0.320629。[配对原始数据](single_video/base.json) · [Dense 误差](dense_error.json)

base 入口和相应常驻 worker 已默认启用，可用 `--no-aligned-coefficients` 单独还原。验证中四种窗口均命中预期的 IR 布局。GPU6 正式入口 base、同主干 Dense 各预热后运行 5 次，中位数 5.05521 s、6.50831 s，加速 1.28745×，距 1.40034× 墙钟目标还有 407.54 ms。两者输出分别与上一版归档逐位一致。[正式入口复测](deployment.json)
