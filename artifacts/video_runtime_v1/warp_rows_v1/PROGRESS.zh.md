# SM80 融合注意力的精算页预取

P432 每次完整融合调用有约 158.15 GFLOP 主矩阵工作，其中精算 QK 和双项 AV 占 73.9%。原 TTGIR 中这三次 dot 共用列方向的四个 warp。每个精算页只有一份 K 和一份 V 共享缓冲，原循环在页尾发起下页两次拷贝并一起等待。实验只调整锚点正常分支，保留首尾页处理、异常回退、分母、双项 BF16 AV 和补偿顺序。[原始 IR](control_432.ttgir) · [新 IR](prefetch_kvwait_432.ttgir)

新循环在当前 V 片段进入寄存器后即发出下一页 K 拷贝，在当前页 AV 后发出下一页 V 拷贝。页尾等待 K，下一轮在第一次读取 V 共享缓冲前等 V。A100、Q64、BR64、四个 warp、两级流水保持一致。修改作用于分配物理共享内存和插入 CTA 屏障之前。P432 的单 CTA 遍历 43 个精算页，Compute Sanitizer Racecheck 报告 0 hazards。极端输入让第二页压过首页，触发在线回退，两版输出有限且相等。[Racecheck](kvwait_racecheck.txt) · [极端输入](stress.json)

## 完整融合核

三种较小窗口使用同一真实输入的前缀，选页 ID 截短并对页数取模。每种窗口单独测当前 JIT 与候选，CUDA Graph 预热后测 20 对，每次 8 个 replay。表中时间为各自中位数，加速取配对比值中位数。四个窗口输出均完全相同。[测试程序](measure.py)

| 总页数 | 当前 ms | 新 ms | 加速 |
|---:|---:|---:|---:|
| 216 | 0.61843 | 0.61453 | 1.00596× |
| 288 | 0.77395 | 0.76698 | 1.00927× |
| 360 | 0.99974 | 0.98874 | 1.01120× |
| 432 | 1.14464 | 1.12800 | 1.01487× |

按 116、116、116、464 次调用加权，设备时间估算从 808.60 降到 798.34 ms，节省 10.26 ms。Dense 有效吞吐折算参考 528.33 ms。满窗口实测主项有效吞吐从 138.2 提升到 140.2 TFLOP/s。[工作量与 Dense 口径](../compensation_scale_v1/attention_budget.json)

## 版式和等待位置

P432 原 IR 重编译对照为 0.9997×。将 QK 和 AV 的 warp 布局从 1×4 改成 2×2 为 0.898×，4×1 为 0.684×。4×1 配合概率片段直接转换为 0.795×。将 Q 保留在寄存器为 0.676–0.688×。单独提前 K 为 0.946×，单独拆开等待为 0.988×。K/V 各使用两个槽位，共享内存升到 96 KiB，两种等待策略为 0.737× 和 0.713×。将所选方案的页索引再提前曾测到 1.0165×，两者的微小差异没有做配对验收，本轮部署采用先完成四窗口验证的调度。

| P432 Nsight Compute | 原 | 新 |
|---|---:|---:|
| 每线程寄存器 / 每 CTA 共享内存 | 255 / 80 KiB | 255 / 80 KiB |
| 编译 spill 槽 | 72 | 80 |
| local load sectors | 5,543,424 | 1,105,920 |
| Long scoreboard stall | 12.86% | 7.46% |
| Short scoreboard stall | 13.33% | 19.39% |
| Tensor pipe active | 45.10% | 45.66% |
| Achieved occupancy | 12.27% | 12.13% |

这组计数说明静态 spill 槽数不能代表执行中的 local load 流量。长依赖等待下降，短依赖等待上升，tensor pipe 利用率仅提高 0.56 个百分点。后续值得在源指令级拆分概率片段的共享写回和读取，以及 AV 的依赖等待，确认新增的短等待落在哪些指令上。[原硬件计数](control_ncu.csv) · [新硬件计数](prefetch_kvwait_ncu.csv)

## 视频与部署

105 帧、9 chunks，交替运行顺序，排除冷启动。原型同视频 8 对中 7 对更快，配对中位节省 9.97 ms。5 个 prompt × 2 个 seed 的 10 对全部更快，配对中位节省 11.32 ms。原型用直接启动已编译核的路径。接回正常 Triton JIT 分发后，四种特化都命中新调度，同视频 8 对全部更快，配对中位节省 11.02 ms，1.00218×。所有比较的九个 chunk latent 均逐位相等，计数一致。city_walk、seed 701 相对同主干 Dense 的 latent NMSE 两版均为 0.378409，MAE 均为 0.320629。[单视频原型](prefetch_kvwait/single_video/base.json) · [十组验证](prefetch_kvwait/multi_video/base.json) · [正常启动路径](release/single_video/base.json) · [Dense 误差](dense_error.json)

默认对 base 启用，可用 `--no-page-prefetch` 复测旧路径。实现复用 Triton JIT 的设备、签名与对齐特化，按原编译缓存键和新 IR 内容缓存产物。仅在 Triton 3.2.0、SM80 和已验证的循环结构下替换调度。首次编译的耗时单独计入冷启动。[部署源码](../../../local_experiments/video_runtime_v1/page_prefetch.py)

GPU5 独立采集完整视频时间线，旧、新融合核累计分别为 808.57 和 797.19 ms，实际收回 11.38 ms。新核按 109.93 TFLOP/视频计算达到 137.9 TFLOP/s，距 528.33 ms 参考仍差 268.86 ms。GPU7 正式入口先后测 base 与共享主干原生 Dense，各预热后取 5 次中位数，分别为 5.03268 s、6.46920 s，加速 1.28544×。按 1.40034× 验收线计算，仍差 412.94 ms。两个正式输出分别与上一版 base、Dense 的归档 latent 逐位相同。[旧时间线](timeline_off.json) · [新时间线](timeline_on.json) · [部署复测](deployment.json)
