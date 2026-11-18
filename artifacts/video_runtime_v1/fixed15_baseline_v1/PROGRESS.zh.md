# 首帧锚点、固定 15% 精算预算的 base

首个 latent 帧的 1536 个 token 占 24 页，在历史环形缓存中常驻。进入稀疏区间后，每个 Q64 首先选自己的页，再取 24 个锚点页，按原 Risk 排序补齐至 ⌊0.15P⌋。216 / 288 / 360 / 432 页分别精算 32 / 43 / 54 / 64 页。前两段、原生层及 clean commit 继续走 Dense。base 和 worker 的 c_risk 使用这一口径，历史 10% 对照可用 accepted_tiers 的 --no-first-frame-fixed15。

30 秒 horse_field / 2027 使用已保存的相同输入重放，41 个 chunk。正式路由输出的视频与用户选定的视频 SHA-256 相同（2d8c6091090a9002e6e3f1dc13ab707bea7be2ef841f3545cbd7a8aeced2dd86）。10% 专用的预取和系数布局由页数签名分流。15% 走正式 _tail，开启 anchor_scale、column_mass、guard 和双项 BF16 AV。

后续已将 15% 形状接入预取和补偿布局，见[最新调度验收](../fixed15_prefix_v1/PROGRESS.zh.md)。以下保留此次算法基线建立时的测量。

## 105 帧重测

A100-SXM4-40GB，city_walk / 701，512×768，9 个 chunk。各方法完整预热后，在同一卡上各测 5 次生成到 latent 的墙钟，中位数 base 5.33890 s，原生同主干 Dense 6.47041 s，加速 1.21194×。新 base 相对该 Dense 的 latent NMSE 为 0.55755，余弦 0.71304。Dense 参照沿用原生缓存策略。此距离记录算法差异，仅代表该样本。

| 窗口 | 精算页 | 次数 | 视频时间线每次 ms | 按同窗 Dense 吞吐折算 ms |
|---:|---:|---:|---:|---:|
| 216 | 32 | 116 | 0.803 | 0.520 |
| 288 | 43 | 116 | 1.004 | 0.700 |
| 360 | 54 | 116 | 1.270 | 0.881 |
| 432 | 64 | 464 | 1.468 | 1.040 |

融合核 150.289 TFLOP，1038.08 ms，144.78 TFLOP/s；逐 chunk 用 Dense 减保留 FlashAttention 得到被替换部分的吞吐，加权约 206.91 TFLOP/s，折算 726.34 ms。设备累计差 311.74 ms，需要 1.429× 融合核加速。满窗口贡献约 199 ms 差额。主项含精算 QK、双项 AV、代表 QK、均值双项 AV 与按 16 列填充的 detail 双项 AV。路由、归约、投影重建以及数据搬运的耗时均进入相应的实测，融合主矩阵 FLOP 只计上述矩阵工作。

按 5×30 个层调用、前两段全 Dense、后七段各保留 34 个 Dense 调用重新估算整条视频。沿用每个稀疏层路由 4–5 GFLOP 规划值，同吞吐主项参照为 4.527–4.531 s（1.428–1.429×）。路由估算没有逐算子覆盖 top-k，模型也略去 summary/commit、访存和发射成本。该参照与上表的逐窗口硬件吞吐参照分别计算。

设备时间线主要模块为其他 BF16 GEMM 1356.45 ms、保留原生自注意力 1102.23 ms、融合注意力 1038.08 ms、FFN 第一层及 GELU 742.92 ms。独立的阶段插桩测得 attention 2031.90 ms / 1305 次、route_and_correction 316.55 ms / 812 次、summary_and_encoding 127.51 ms / 812 次、clean_commit 53.02 ms / 261 次。阶段与设备核统计存在包含关系。

P432（64 精算页）NCU 实测 HBM 吞吐占比 6.19%，Tensor pipe active 48.01%，short / long scoreboard 13.09% / 12.49%，活跃 warp 占比 12.27%。平均活跃 CTA 为 1.96/SM，每 CTA 4 warp，约 7.85 活跃 warp/SM。内存带宽有余量；更改 CTA 大小本身不能据此推断延迟隐藏提升。旧 43 页的 PC 等待占比不用于这份结论。

## 两项结构实验

- **双查询块**。在一个 CTA 使用 8 个 warp，精算用 3D batched dot 读取独立页表，补偿共享 128 行的摘要分块，按查询块独立屏蔽精算页。在 Triton 3.2.0 上双 stage 的 LLVM lowering 崩溃。单 stage 可以直接编译，默认 batched MMA 布局为 [8,1,1]，耗时 225.20 ms。显式指定 [2,1,4] 并经 shared 交接概率后，P432 完整融合结果与正式 base **逐位相同**。两者配对中位数 2.34822 对 1.46125 ms，加速比 **0.6223×**。资源为 255 寄存器/线程，140 spill 槽，128 KiB shared/CTA；对照为 255、72、80 KiB。新 CTA 的共享内存只允许单 CTA 驻留。3.3.0 另作编译诊断，不计入部署对照。
- **手写 CUDA SM80 精算**。每 warp 取 16 行，概率在寄存器中转成 MMA 的 A 操作数，V 用 ldmatrix.trans 读出，两个输出片段复用 8 个 FP32 临时分量。对齐了双项 AV 运算次序、BF16 转换、分母归约树、除法和 log2。P432 真实输入全部行通过现有 anchor 质量条件，FP32 精算输出、LSE、质量逐位一致。先测逐页搬运 1.05683 ms，再把下一页 K 拷贝与当前 AV 重叠，得到 **1.03866 ms**。与同一轮 Triton 精算 0.97357 ms 配对，加速比 **0.9364×**。优化后的 ptxas 报告 255 寄存器、32 KiB shared、零 spill。guard 回退路径和完整 CUDA 补偿尚未实现，此处比较的是同一正常精算阶段。源码及资源记录见 warp_exact.cu、cuda_resources.txt。

两个候选均未通过模块速度门槛，未接入生成或启动多视频验收，本轮生产配置仅变更算法基线。

## 复测入口

运行时从仓库根目录设置 PYTHONPATH=.。capture.py 从已保存的视频输入提取四个窗口；bench.py P --baseline-only 计时正式融合核。模块计时预热 60 次，使用 CUDA Graph，每个样本重放 8 次，共 20 组，成对测试交替顺序。执行 bench.py 432 --paired-stages 1 后，layout_test.py 432 --paired-stages 1 可复测显式双查询块布局。CUDA 用 nvcc -arch=sm_80 -O3 --fmad=false -lineinfo -Xptxas=-v -Xcompiler=-fPIC -shared warp_exact.cu -o /tmp/jzk_fixed15_exact.so 编译，再运行 cuda_gate.py 432。

原始墙钟见 base_release.json、dense_release.json；逐窗设备事件见 base_timeline.json、dense_timeline.json；模块和逐核结果见 module_profile.csv、kernel_profile.csv；预算脚本 budget.py 生成 budget.json。baseline_ncu.csv 和 baseline_{P}.json 分别记录硬件计数和真实输入的隔离计时。
