# 第三轮：补偿分支融合

按照 `/home/shuchenweng/jzk/AGENTS.md` 执行 profile、理论算术量核对、模块输入输出检查和模块计时，再进行单视频多模块验收。GPU6/7，city_walk / seed701 / 9 chunks。固定全覆盖 C seed17、旧配套 E48 Full_Pset Reader、E27 Phi、B16，未混入新训练 Reader。

## 热点及理论计算量

当前整路由融合版本，72 个 head–tile、410 页的 CUDA Graph 隔离计时：路由 2.665 ms，补偿 0.954 ms。profiler 显示路由末端 FP64 动作 kernel 占路由 CUDA 时间约 39%，补偿 bmm 占补偿时间约 54%，归约、逐元素计算和复制构成其余主要开销。

每批主算术量：Q64–R4 QK 1.935 GFLOPs；完整均值 AV 0.484 GFLOPs；rank8 detail 0.091 GFLOPs；Reader 系数组合 0.064 GFLOPs；Phi 组合 0.030 GFLOPs；中心投影 0.045 GFLOPs。这些只统计对应乘加，不把 exp/log/sort 等价成 FLOPs。补偿约 0.575 GFLOPs 的主 AV 算术不足以单独解释接近 1 ms 的耗时，因此本轮融合归一化、mask 和中间计算，减少 kernel 和临时张量。

## 代码与算法

`BatchGraphs` 增加实例级 `compile_tail` 选项，预热后捕获融合补偿。`VideoRuntime(..., compile_route=True, compile_tail=True)` 启用。默认 `compile_tail=False`，原实现保留用于对照。Page64/Q64/R4、完整 128 维均值、rank8、精算页排除和 LSE 合并均未改；无训练或路由算法改动。

新增 `profile_route.py` 保存模块 profiler 表和 Chrome trace、理论计算量；`check_tail_fusion.py` 做同输入模块契约与 CUDA Graph A/B；`benchmark_hotspots --mode tail` 在整路由融合之上开启补偿融合。

## 验证与速度

同进程 CUDA Graph 25 次中位数：1.1305 → 0.8858 ms，1.276×。与最初 profiler 0.954 ms 不混算，速度比使用同次 A/B。

补偿输出相对原实现 NMSE=1.65e-8、最大绝对误差=3.90e-4，LSE 最大误差=9.54e-7。最初更严的 1e-10 NMSE/1e-4 max_abs 门禁未通过，故不声称逐位等价；额外检查完整 Attention，原有验收门禁通过，NMSE=1.83e-7、max_abs=0.0078125。14 个评分案例、主导页移出、Risk 和 no-op 检查通过。补偿隔离契约使用 1e-7 NMSE/1e-3 max_abs，显式作为近似实现测试。

端到端完整预热后各两次：

| 版本 | 两次生成秒 | 中位秒 | Attention profile 秒 | 路由 profile 秒 |
|---|---|---:|---:|---:|
| 整路由融合、原补偿 | 40.0210 / 40.0026 | 40.0118 | 10.3953 | 22.0022 |
| 整路由融合、融合补偿 | 37.8244 / 37.8130 | 37.8187 | 8.2439 | 21.9614 |

端到端 1.058×，耗时减少 5.48%。约 2.15 秒来自 Attention，路由和摘要耗时基本相同。冷运行（包含预热生成及编译/capture）分别 56.98 / 58.58 秒，计时不含模型加载、VAE、文件 IO。当前累计相对最早 80.40 秒版本约 2.13×，该累计值跨轮测量。

闭环 latent 相对本轮基线 NMSE=0.19442、max_abs=4.109375，差异明显。局部误差通过 Attention 门禁不能替代视频质量验证，因此此数值近似优化保留为显式实验选项，不提升为默认路径。没有新增视觉质量改善结论。

## 下一热点

更新 profile：路由 21.96 秒，Attention 8.24 秒。路由末端 FP64 动作归约仍值得优先处理；下一轮应拆分其昂贵指数/归约与数据布局，保持微小正分和主导页语义。

所有原始结果和输入权重/代码 SHA：`artifacts/video_runtime_v1/hotspot_round3/`，其中 `profile/` 为算子表与 trace，`tail_fusion.json` 为模块 A/B，`validation/` 为原门禁，`e2e_before/`、`e2e_tail/` 为完整计时、latent 和 profile，`summary.json` 为汇总。测试中的多个 head–tile 来源于同一个视频事件，不是多视频 batch。
