# 分组排序、补偿输入与 Risk 上界

上一部署 base 时间线中精算占 534 ms，补偿占 352 ms。满窗口精算的三项矩阵乘约 116.87 GFLOP，0.772 ms 对应约 151 TFLOP/s。针对页读取、本地排序和摘要拷贝分别做了隔离测试。

## 精算页缓存

同一真实满窗口 Q/K/V 及页表，缩短窗口时取前缀并对页号取模。每组 30 对计时，各次回放十遍。比较 K 页内转置、按 16 维分块以及原 128 维连续排列，并分别测试 V 原布局和同页同头连续布局。六种配置在四个窗口的 FP32 输出与 LSE 全部一致。K 和 V 均按页、头、token、通道连续排列时，P216/288/360/432 的精算分别加速 1.073/1.044/1.045/1.049×。整段生成还要写入当前缓存，clean 步的 Dense attention 要恢复 token-major 输入。105 帧 base 实测 8 对，每对变慢 110–130 ms，配对中位回退 116 ms。该运行时原型已移出默认路径。[隔离计时](../exact_page_layout_v1/k128_v1.json) · [单视频](../exact_page_layout_v1/base.json) · [实验实现补丁](../exact_page_layout_v1/runtime.patch)

## 分组排序

旧分组每一级分别生成 int64 排序索引、收集 token id，并在下一层还需使用特征时收集 32 维 FP32 特征。每层处理 12 × 4608 个 token，单级特征约 6.75 MiB。32–512 token 的小组用一个 Triton 块完成稳定排序与收集，保留相同分数的原顺序。1536/4608 token 的大组分别用 CUB 256/512 线程做稳定基数排序，另一个核并行搬运特征和 id。测试含重复值、正负零及 NaN。

小组的排序与收集加速 1.92–3.18×，大组加速 2.34/3.37×。大组直接使用 Triton bitonic 排序分别只有 0.70/0.24×。四流完整分组按每层计，小组融合 0.9404→0.8797 ms；在独立配对中加入大组基数排序再从 0.8821 降至 0.8368 ms。四个输入的分组 K、V 和顺序逐项相同。[小组](../group_sort_v1/module.json) · [大组](../group_sort_v1/radix_module.json) · [完整分组](../group_sort_v1/radix_grouping.json)

## 补偿与路由

补偿核直接按每个头的物理页间距寻址。P216/288/360 每次省去 12 × P × (4×128+128+24) 个 BF16 的打包，完整 attention 快 1.5%–1.8%。满窗原本已连续，速度持平。单视频每档 12 对的配对中位仅节省 5–8 ms，波动中出现回退。追加五条视频，每条每档两对且顺序相反，30 对中 28 对更快，全部 latent 字节一致。base 有一对旧版计时偏高导致表观节省 469 ms，保留原始记录，base 的十对中位节省 12.4 ms。[模块](../strided_tail_v1/module.json) · [多视频 base](../strided_tail_v1/multi_video/base.json) · [intersection](../strided_tail_v1/multi_video/intersection.json) · [step](../strided_tail_v1/multi_video/step.json)

Risk 的每头每页系数由一个核读取页统计量、计算头内最大值并加回各页。每头只需 P 次 BF16 读取和 P 次 FP32 写入。沿用已验收的 softmax 分支选择，满窗完整 Risk 从 0.2904 降至 0.2636 ms，四个窗口分别加速 1.037/1.093/1.096/1.102×。Risk 和选页 mask 均相等。计时脚本额外检查 profiler 中候选调用了新核。首次用 partial 区分编译函数时两个 profile 都走了旧核，已换成两个独立入口重新验收。[有效模块测试](../risk_bound_v1/module.json) · [新核 profile](../risk_bound_v1/new_profile.json)

另测 Risk 主归约的大页块及按页映射相邻线程的 CUDA 核。满窗 32 页/8 warp 约 0.186 ms，对照为 0.180 ms。其余大页块更慢。CUDA 页映射约 0.314 ms，最大绝对差约 3.8e-6，保留原主归约。[页块对照](../risk_tiles_v1/module.json) · [CUDA 对照](../risk_tiles_v1/cuda_pages.json)

## 合并后的单视频验收

city_walk，种子 701，105 帧，只计生成。每档预热后新旧顺序交替 8 对。

| 档位 | 旧中位/s | 新中位/s | 加速 | 每对节省范围/ms | latent 与路由计数 |
|---|---:|---:|---:|---:|---|
| [base](base.json) | 5.42902 | 5.39523 | 1.0063× | 24.7–37.7 | 一致 |
| [intersection](intersection.json) | 5.42750 | 5.38809 | 1.0073× | 31.5–50.3 | 一致 |
| [step](step.json) | 6.16625 | 6.12366 | 1.0070× | 36.6–56.6 | 一致 |

## 当前部署

三档正式入口分别预热后计时五次，核实新排序、按摘要缓存步长寻址、匹配归约分支的 Risk 上界均启用。base、intersection、step 的生成耗时中位数分别为 5.35537、5.46413、6.09852 秒，最终 latent 与上一部署逐字节一致。GPU7 同主干原生 Dense 仍沿用 6.58614 秒，当前 base 加速 1.22982×，距目标 4.70324 秒尚差 0.65213 秒。[默认入口](deployment.json) · [GPU 时间线](device_timeline.json)

新时间线中精算、补偿、Risk 行归一化分别为 531、357、66 ms。P432 的 12×1152×1728 个 FP32 路由 logits 占约 91 MiB，后续将据此评估 QK 生产和归一化之间的读写成本。
