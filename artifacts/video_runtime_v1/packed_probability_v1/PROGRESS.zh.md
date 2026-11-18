# 精算注意力的 BF16 概率转换

105 帧 base 时间线中，融合核 `_tail` 的设备时间累计 852.9 ms。432 页输入的矩阵乘主项约 158 GFLOPs，实测约 130 TFLOP/s。此前同形状硬件计数显示 Tensor Core 活跃率为峰值的 42.14%，HBM 吞吐率为峰值的 10.41%。本轮检查每页 QK 与双项 AV 之间串行执行的概率转换。[时间线](../repeated_context_v1/packed/device_timeline.json) · [硬件计数](../tail_boundary_v1/PROGRESS.zh.md)

保留 FP32 残差和两次 AV 乘加，将相邻两个概率成对舍入为 BF16，并重排高低项。完整内核 PTX 中，标量 FP32→BF16 指令从 214 条降到 150 条，增加 32 条成对转换和 32 条字节重排。BF16 MMA 指令仍为 300 条。这些数字是静态指令数。[原始 PTX 计数](exact_old_ptx.json) · [候选 PTX 计数](exact_new_ptx.json)

432 页使用真实捕获输入，较小窗口截取同一输入并对页号取模。各窗口用 CUDA Graph 预热后交替测 16 对，每个计时包含 8 次回放。表中加速比为成对加速比的中位数。四种窗口在两轮测试中输出均相等。[首次测量](exact.json) · [跨卡复测](exact_repeat.json)

| 页数 | 基线 ms | 成对转换 ms | 加速比 | 跨卡复测 |
|---:|---:|---:|---:|---:|
| 432 | 1.21216 | 1.20672 | 1.00467× | 1.00430× |
| 360 | 1.06490 | 1.04922 | 1.01471× | 1.01411× |
| 288 | 0.81920 | 0.81869 | 1.00070× | 1.00055× |
| 216 | 0.65414 | 0.64915 | 1.00759× | 1.00692× |

另外比较了拆分 QK 和 AV、堆叠高低项、移动最大值归约以及尾部概率转换。QK 按 64 通道拆分仅有 1.003–1.006×；按输出通道拆分 AV 的速度为基线的 0.827–0.848×。堆叠高低项也未改善模块速度。[QK](qk64.json) · [AV](av64.json) · [行堆叠](stackrow.json) · [分半堆叠](stackhalf.json)

视频使用 105 帧，关闭 matmul 和 cuDNN 的 TF32，预热后交替执行基线和候选。单视频使用 city_walk / 701。多视频仅在 base 上使用 kitchen_cook、forest_run、portrait_turn、reef_fish、tennis_match，各跑 703 和 704 两个 seed。配对节省为同一输入上基线耗时减去候选耗时。

| 验收 | 卡 | 基线 / 候选中位数（s） | 成对加速比中位数 | 成对节省中位数（ms） | 候选更快 |
|---|---:|---:|---:|---:|---:|
| [single_video/base](exact/single_video/base.json) | 6 | 5.14108 / 5.14043 | 1.000432× | 2.22 | 6/8 |
| [single_video/intersection](exact/single_video/intersection.json) | 4 | 5.17416 / 5.17279 | 1.000266× | 1.38 | 3/5 |
| [single_video/step](exact/single_video/step.json) | 5 | 5.85969 / 5.86310 | 1.000284× | 1.67 | 3/5 |
| [multi_video/base](exact/multi_video/base.json) | 7 | 5.10782 / 5.10839 | 1.000182× | 0.93 | 5/10 |

全部 28 对最终 latent 逐字节一致，最大绝对误差为 0，路由计数逐对相同。多视频中候选快 5 次、慢 5 次，成对节省从 −4.22 到 +14.17 ms。基线先跑时节省中位数为 −2.46 ms，候选先跑时为 +2.46 ms。端到端收益未能稳定复现，候选仅保留在实验目录。最近部署仍为 base 5.10156 s / Dense 6.47587 s，即 1.26939×。

复现入口为 [模块计时](measure.py)、[候选内核生成](probability.py)和[视频配对](video.py)。在仓库根目录设置 PYTHONPATH=.，运行本目录的 measure.py exact 或 video.py exact --tier base --multi-video。
