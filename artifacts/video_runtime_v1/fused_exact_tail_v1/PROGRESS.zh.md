# 单 kernel 页精算、选页掩码与补偿

原部署在 105 帧 base 的精算和补偿分别累计 533.6、358.1 ms。P432 每次有 43 个精算页，双项 BF16 AV 加 QK 约 116.87 GFLOPs；未选页补偿主矩阵乘约 41.28 GFLOPs。两段在同一 head、同一 64 行查询上顺序消费数据，可以省去中间输出的往返。[上轮设备时间线](../rechecked_mass_v1/device_timeline.json)

融合路径从精算直接返回 FP32 输出和 LSE，在同一 CTA 内生成 16 个页掩码字，随后沿用现有补偿和归一化公式。精算原本由编译器收缩的两处乘加在融合核中显式使用 FP32 FMA。每次调用省去 57,065,472 字节的显式 FP32 输出与 LSE 读写，以及两次 kernel 启动。四个历史窗口中，独立精算回退与改动前的 CUDA PTX 指令序列一致，真实输入的精算输出和 LSE 逐位一致。[回退核测试](baseline_compat.json)

## 模块与同视频验收

组合模块包括页精算、掩码、补偿、尾部合并；使用已捕获的 layer15/chunk8 Q/K/V，其他窗口用相同输入和取模页 ID 比较。每种窗口预热后交替测 24 对，每次图回放 10 遍。两边的 BF16 输出逐字节一致。[模块原始样本](module.json)

| 历史页 | 原模块（ms） | 融合（ms） | 加速比 |
|---:|---:|---:|---:|
| 216 | 0.7108 | 0.6996 | 1.0160× |
| 288 | 0.8622 | 0.8218 | 1.0491× |
| 360 | 1.0982 | 1.0655 | 1.0307× |
| 432 | 1.2626 | 1.2144 | 1.0397× |

city_walk、seed 701、105 帧，按档位在同一卡交替运行 10 对，预热另计。30 对全部提速，所有最终 latent 逐字节一致，各档路由计数一致。

| 档位 | 原中位数（s） | 融合中位数（s） | 加速比 | 每对节省（ms） |
|---|---:|---:|---:|---:|
| [base](base.json) | 5.37244 | 5.33919 | 1.0062× | 10.3–41.3 |
| [intersection](intersection.json) | 5.41463 | 5.38177 | 1.0061× | 20.1–50.9 |
| [step](step.json) | 6.06570 | 6.03252 | 1.0055× | 17.9–38.6 |

## 硬件计数

P432 隔离事件，Nsight Compute 的旧精算、掩码、补偿合计 1.584 ms，融合约 1.538 ms。两边测得 DRAM 读约 234.6 MB 和 200.3 MB，DRAM 写约 29.9 MB 和 39.0 MB（融合取首次采样）。编译资源从原精算 252 寄存器、0 spill 和原补偿 255 寄存器、0 spill，变成融合核 255 寄存器、74 个 spill 槽、80 KiB 共享内存，实测 occupancy 为 12.26%。因此 57.1 MB 仅指省掉的显式中间读写。手动将精算向量写入 CTA 独占全局缓冲并重新加载，P432 仅 1.0045×，慢于自动分配。[原核计数](ncu_old.csv) · [融合核计数](ncu_fused.csv) · [编译资源](resources.json) · [手动缓冲](../exact_scratch_v1/manual_spill.json)

## 变体与复现

三种中间结果布局在 P432 为 0.932–0.990×。保留单独掩码的融合版是 1.005×；先算补偿再精算是 0.974×。本轮单 kernel 采用 4 warps、2 stages。[布局与融合变体](../exact_scratch_v1) · [精算首轮与多视频复核](../exact_page_load_v1/PROGRESS.zh.md) · [补偿系数与误差](../tail_butterfly_v1/PROGRESS.zh.md)

模块复现使用 `python -m local_experiments.video_runtime_v1.check_tail_merge --fused-exact-tail`。单视频交替比较使用 `python -m local_experiments.video_runtime_v1.check_rms_row_video --tier base --fused-exact-tail --pairs 10`，另两档替换 tier。默认入口为 accepted_tiers，使用 `--no-fused-exact-tail` 可复现对照。

默认入口预热后各测五次。base、intersection、step 中位数分别为 5.31956、5.38010、6.02160 秒，最终 latent 与上一部署逐字节相同。沿用 GPU7 同主干原生 Dense 的 6.58614 秒，base 加速 1.23810×，距离 4.70324 秒的目标还差 0.61632 秒。新时间线中原独立精算和掩码均已消失，融合核累计 853.2 ms，旧三核累计 894.9 ms。[正式入口](deployment.json) · [设备时间线](device_timeline.json)
