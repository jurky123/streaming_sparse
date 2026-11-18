# 文本交叉注意力合并重复位置

city_walk/701 的文本嵌入有 512 个位置，从索引 38 起完全相同。当前原生 PyTorch 2.6 FlashAttention 使用 64 个键一块，并从后向前遍历。若投影后第 63 至 511 行的 K、V 分别逐位相同，最后七块都只有同一个键和值。第一次遍历的这 448 个位置给每行贡献 448 单位的未归一化概率质量，分子为 448 V。运行时从该状态开始，再处理前 64 个位置。每次交叉注意力的 QK 和 AV 主项从 14.50 GFLOPs 降到 1.81 GFLOPs。

实现保留原生内核的 BF16 概率舍入、FP32 累加、四个 lane 的归约顺序及两次独立的乘法舍入。层缓存初始化时核对投影后 K/V，匹配时将前 64 行排成 MMA 操作数的顺序。每层增加 396288 字节，30 层合计 11.34 MiB。其他文本长度、K/V 后缀、设备条件及显式上下文长度沿用原路径。开启整块图捕获的配置沿用原路径。

第 0、12、29 层真实输入上，模块加速 2.117–2.155×，每层 7077888 个 BF16 输出逐字节一致。模块测量使用已打包的 K/V；视频计时包含每条视频重新检查和打包。分别破坏 K 或 V 重复条件、改变长度以及禁用后重新启用均通过检查。[隔离模块](compact_pack_module.json) · [缓存条件](cache.json)

| 单视频 city_walk，seed 701 | GPU | 对数 | 旧 / 新中位数（秒） | 加速比 | 成对节省中位数（ms） |
|---|---:|---:|---:|---:|---:|
| [base](packed/base.json) | 7 | 5 | 5.15229 / 5.11094 | 1.00809× | 46.65 |
| [intersection](packed/intersection.json) | 4 | 3 | 5.23383 / 5.17210 | 1.01194× | 46.46 |
| [step](packed/step.json) | 5 | 3 | 5.90999 / 5.85019 | 1.01022× | 45.55 |
| [dense_gpu6](packed/dense_gpu6.json) | 6 | 5 | 6.56513 / 6.51973 | 1.00696× | 44.67 |
| [dense](packed/dense.json) | 7 | 5 | 6.52132 / 6.47398 | 1.00731× | 50.60 |

三档稀疏与原生 Dense 均默认启用，--no-repeated-context 可在稀疏入口关闭。单视频已覆盖收益并验证全部 latent 与计数。GPU7 使用相同主干分别独立预热后测 5 次，base 5.10156 秒、Dense 6.47587 秒，加速 1.26939×。两者最终 latent 各自与上一部署逐字节一致。按 1.40034× 验收线，目标 4.62450 秒，还差 0.47706 秒。两端都获得加速，目标时间随本轮 Dense 复测更新。[部署](packed/deployment.json) · [新时间线](packed/device_timeline.json)

原型中，直接折叠的 Triton 版在三个模块样本上仍有 285–567 个 BF16 元素不同，单视频 NMSE 0.384。更接近原生归约的 Triton 版本仍有 16–23 个元素不同，视频 NMSE 0.277，速度与基线接近。两者均仅作为实验。手写 CUDA 的第一次实现还有舍入差异，在缩放得分与缩放最大值两边均显式使用 __fmul_rn 后消除。按 MMA 分片打包 K/V 后，原生等值路径获得上述模块与单视频收益。[Triton 模块](verified_module.json) · [快速原型视频](base_plain.json) · [归约原型视频](base.json) · [CUDA 首版](compact_first.json)

复现时从仓库根目录运行 python -m local_experiments.video_runtime_v1.check_rms_row_video --repeated-context --tier base --pairs 5。隔离模块的真实输入可用本目录 measure.py 生成到 /tmp/jzk_cross_three.pt。用于对照上游实现的实验构建脚本为 native_reference/build.py，当前默认核使用仓库自带的 csrc/repeated_context.cu。
