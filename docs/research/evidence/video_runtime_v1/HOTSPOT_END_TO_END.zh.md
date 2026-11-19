# 单视频完整生成与下一轮路由优化

输入 city_walk / seed701 / 9 chunks；同一全覆盖 C seed17 deployment、配套旧 E48 Full_Pset Reader、E27 Phi、B16。新 Full_Pset 训练权重未混入此次对照。GPU6/7 为相同型号 A100。每个版本先完整生成一次预热，再计时两次，另一次 profile 记录 CUDA Events；排除模型加载、VAE、文件 IO。各版本冷启动单独保存在 result.json。

| 实现 | 稳定生成秒 | 路由修正秒（独立 profile） | Attention 秒 |
|---|---:|---:|---:|
| 原 grouped scorer | 80.404 | 62.348 | 10.417 |
| fused scorer | 44.256 | 26.257 | 10.386 |
| fused 整条路由 | 40.006 | 22.000 | 10.381 |

第一轮端到端 1.817×；第二轮额外 1.106×；累计 2.010×。每组仅两次计时，且使用不同物理卡，不能据此宣称普适速度。路由仍是最大热点，其次 Attention，下一轮可进一步拆分路由中的矩阵乘、排序和读出。

第二轮对 BatchGraphs.compute 全图编译，将共享 logits、Risk、Reader、候选、评分、动作/LUT 合并优化；保留原配置和 FP64 评分。新入口 `VideoRuntime(..., compile_route=True)`，默认关闭该额外选项以便显式选择和回退；benchmark_hotspots --mode route 使用此入口。编译在 graph 预热中进行，内部 cudagraphs 关闭。

四个页面形状 216/288/360/410 的首个事件分别对照，mask/LUT 相同，logits 逐位相同，Risk 最大绝对差 1.53e-5（相对容差门禁通过）；这只是每种形状的抽样，不是全部事件动作一致证明。单独评分器的14例微小正分和主导页数值门禁已通过。

闭环 latent 不逐位一致：fused scorer 相对基线 NMSE=0.03005、max_abs=3.171875；整路由融合 NMSE=0.03194、max_abs=3.2109375。微小浮点/动作边界差异可能在闭环累积，但尚未定位具体差异源；不能将局部门禁视为视频质量等价。未做新增多视频质量测试，也未声称视觉改善。

原始计时、权重/代码 SHA、latent、阶段时间、四形状检查见 `artifacts/video_runtime_v1/hotspot_e2e/`；可重复入口 `python -m local_experiments.video_runtime_v1.benchmark_hotspots --mode baseline|fused|route --output <fresh-directory>`。性能计时仍限定同一个视频。
