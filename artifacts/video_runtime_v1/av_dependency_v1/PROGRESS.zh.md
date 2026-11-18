# SM80 满窗口 AV 操作数依赖与 warp 内概率传递

本轮对评审对应的 page_tail 和工作区已启用补偿布局的 coeff_tail，分别通过正式 JIT 启动 P432。两个核都保留预取和双项 BF16 AV，输出均与原核逐位一致。各自预热 50 次，采集一次完整网格的 SourceCounters，同时导出该次启动使用的 cubin。[采集程序](capture.py) · [预取版 NCU](page.ncu-rep) · [现部署 NCU](coeff.ncu-rep)

## 从等待的消费者追踪生产者

将 NCU 的绝对 PC 对齐至 SASS 函数偏移，按实际回跳边界截取正常精算循环和补偿循环。对 HMMA 的 A、B、C 及 FADD 输入回溯寄存器定义，同时检查 SM80 指令的等待掩码及其绑定的依赖槽。下表为每个采样核独立统计的样本数。采样归属反映消费者当时的等待，不能将样本百分比解释成可回收的运行时间。[归因程序](dependencies.py) · [预取版逐 PC](page_dependencies.json) · [现部署逐 PC](coeff_dependencies.json)

| P432 正常循环 | 预取版 | 现部署 |
|---|---:|---:|
| 精算循环 short scoreboard 样本合计 | 19,060 | 19,625 |
| 其中精算双项 AV 的 HMMA 消费者 | 15,139 | 15,657 |
| 上一行中只等待 A 片段 LDSM 的样本 | 15,139 | 15,657 |
| 精算双项 AV 的 HMMA 上的 stall_wait 样本 | 16,422 | 16,498 |
| 精算 HMMA 结果进入 FADD 的消费者上的 stall_wait | 1,389 | 1,361 |
| 补偿循环 short scoreboard 合计 | 11,377 | 8,629 |
| 其中补偿双项 AV 的 HMMA 上只等待 A 片段 LDSM | 531 | 189 |

预取版的精算循环是 0x41e0–0x6de0。0x5e80 的 HMMA 有 818 个 short 样本，仅等待槽 1。该槽由 0x5e60 的 LDSM 写入 A 所在的 R104；同一 HMMA 的 V 来自更早的 0x4a80。当前 base 的对应例子在 0x6370，有 783 个 short 样本，等待掩码 0x2 指向 0x6360 的 A 片段，V 来自 0x4be0，C 来自 0x6320 的前一次 HMMA。在补偿循环，0x14fd0 的 HMMA 等待 0x14f80 的 A，B 来自 0x14f70。stall_wait 和 short scoreboard 分列，固定延迟链上同时可能有其他就绪约束。[预取版 SASS](page.sass.gz) · [现部署 SASS](coeff.sass.gz)

## 只旁路本 warp 已持有的概率

在锚点精算页内，用线程号和写入序号标记 32 次 packed word 写入，以单 CTA 读回 128 个线程的全部 32×4 个寄存器字。两种部署均实测得到相同规律：每条高低项 A 的 x4 读取有两个 warp 各自拥有一半，其余两个 warp 全部通过 shared。属于本 warp 的 packed word 同时也属于对应的 lane。生产者仍将所有概率写给其他 warp；消费者用寄存器替换自有字、用 x2 读另外半块，其余 warp 保留 x4。高低项及跨页累加顺序保持。另测同一方案的谓词合并发射写法。[原型与标签检查](fragment.py) · [现部署映射](coeff_mapping.json)

| P432 配对 CUDA Graph；各 20 对 | 基线 ms | 原型 ms | 加速比 | spill 槽 | 输出 |
|---|---:|---:|---:|---:|---|
| 预取版，按 warp 发射 | 1.12646 | 3.29555 | 0.3419× | 80 → 230 | 逐位一致 |
| 现部署，按 warp 发射 | 1.09645 | 3.28435 | 0.3340× | 66 → 228 | 逐位一致 |
| 预取版，合并谓词 | 1.12698 | 6.35270 | 0.1774× | 80 → 350 | 逐位一致 |
| 现部署，合并谓词 | 1.09690 | 6.48864 | 0.1690× | 66 → 348 | 逐位一致 |

当前 base 与按 warp 发射候选在同卡先后采样。共享读取 wavefront 从 78,847,414 降到 76,215,110（−3.34%）；local-load sectors 从 898,560 增到 124,816,896，local-store sectors 从 898,560 增到 84,796,416。全核动态指令从 206.14 M 增到 313.86 M，Tensor pipe active 从 47.05% 降至 16.79%。本 warp 的部分读取得到了消减，但自有片段的延长存活和条件发射成本使性能显著回退。结束此原型，止于模块验收。[现部署配对](coeff_local_432.json) · [候选动态计数，两行为基线与原型](local_hardware.csv)

## 版本分流与部署

将 page_tail 和 coeff_tail 的分流都移到构造处，仅在 Triton 3.2.0 创建自定义 JITFunction。其他版本直接绑定部署文件导入的原始 _tail；只有进入 SM80 自定义编译路径才导入所需的内部编译接口。用独立目录中的实际 Triton 3.3.0 wheel 在新进程验证：两个名字均为 _tail，首次通过 coeff_tail 启动真实 P432 成功，结果与同版本原核逐位一致。3.2.0 则分别验证了两种特化命中且结果正确。[代码](../../../local_experiments/video_runtime_v1/page_prefetch.py) · [冒烟结果](compatibility.json)

正式性能仍沿用上一轮的 774.41 ms 融合核时间线，距 Dense 有效吞吐折算参考 528.33 ms 仍差 246.08 ms。本轮只部署版本分流修复。
