# 交集预算搜索：5 prompt × 4 seed 冻结验证

搜索在 city_walk、seed701/702 上完成，341个不同配置、682个视频。此处固定两版搜索结果，与原交集及按层/head/step选择比较；验证结果不用于重新调参。

验证：kitchen_cook、forest_run、portrait_turn、reef_fish、tennis_match；seed703–706。共140视频，8个常驻GPU进程。9 chunks、105帧、768×512、16FPS、history360、tile72、static_route=True。同一prompt/seed的7种方法在同一GPU上执行，Dense仅生成解码一次，稀疏方法顺序轮换。

LPIPS(AlexNet)/SSIM在256×384计算，PSNR使用原分辨率；指标窗口为帧[21,101)，前后段分别[21,61)、[61,101)。均为相对该样本Dense的保真度。prompt和seed未参加这次策略搜索，不代表未参加模型训练。

## 汇总

| 方法 | 生成/s↓ | 相对Dense加速↑ | LPIPS↓ | 前段LPIPS↓ | 后段LPIPS↓ | SSIM↑ | PSNR/dB↑ | 对原交集胜/平/负 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 8.509 | 1.000× | — | — | — | — | — | — |
| 原交集 | 7.762 | 1.096× | 0.36448 | 0.26330 | 0.46566 | 0.40605 | 13.72157 | 0/20/0 |
| 搜索均值最优 | 7.761 | 1.096× | 0.36673 | 0.26329 | 0.47017 | 0.40355 | 13.67402 | 7/0/13 |
| 搜索双seed均衡 | 7.761 | 1.096× | 0.36727 | 0.26634 | 0.46820 | 0.40492 | 13.64668 | 8/0/12 |
| 按层选择 | 8.358 | 1.018× | 0.36607 | 0.26380 | 0.46835 | 0.40789 | 13.70205 | 8/0/12 |
| 按head选择 | 8.691 | 0.979× | 0.36778 | 0.26361 | 0.47195 | 0.40230 | 13.60031 | 11/0/9 |
| 按step选择 | 8.425 | 1.010× | 0.36377 | 0.26116 | 0.46637 | 0.40787 | 13.70715 | 13/0/7 |

耗时为完成基础/3-head/12-head路由形状预热后，每样本每方法一次生成；排除权重加载、VAE、指标和文件IO。表中速度比为均值之比。这是跨场景配对速度记录，不替代搜索报告中同卡三次重复的正式计时。所有方法共用实现，仅改变Full_Pset执行范围；不是kernel提速。

## 两版搜索结果的迁移

- 搜索均值最优：LPIPS相对原交集变化 +0.002248（+0.617%），7/20配对改善；生成加速1.0001×。
- 搜索双seed均衡：LPIPS相对原交集变化 +0.002789（+0.765%），8/20配对改善；生成加速1.0001×。

两版搜索配置的跨场景均值均未超过原交集，搜索集收益没有迁移到本次验证。因此保留原交集作为低预算方案，不用搜索结果替换默认。按step选择的平均LPIPS最低，但比原交集多耗时约0.662s（8.53%）；它使用25%修正预算，原交集仅1.51%，属于不同成本点。此处报告观察结果，不作统计显著性保证。

## 逐场景LPIPS（各4seed）

| 场景 | 原交集 | 搜索均值最优 | 搜索双seed均衡 | 按层选择 | 按head选择 | 按step选择 |
|---|---:|---:|---:|---:|---:|---:|
| kitchen_cook | 0.36270 | 0.36491 | 0.37314 | 0.35200 | 0.36694 | 0.35686 |
| forest_run | 0.33639 | 0.33390 | 0.32865 | 0.34158 | 0.34230 | 0.33364 |
| portrait_turn | 0.31750 | 0.32154 | 0.31406 | 0.31573 | 0.32054 | 0.30948 |
| reef_fish | 0.36717 | 0.36619 | 0.37206 | 0.37377 | 0.36725 | 0.36814 |
| tennis_match | 0.43865 | 0.44710 | 0.44844 | 0.44728 | 0.44187 | 0.45073 |

## 冻结配置与实现

所有交集均为7层×3head×1step=21/1392单元，预算1.51%；按层7/29=24.14%，按head和step各25%。层0不参与，noisy step从0编号。

- 原交集：layers=[5, 14, 27, 3, 8, 17, 4]，heads=[4, 3, 9]，steps=[0]。
- 搜索均值最优：layers=[3, 4, 8, 14, 17, 25, 27]，heads=[3, 4, 9]，steps=[0]。
- 搜索双seed均衡：layers=[3, 8, 14, 17, 25, 26, 27]，heads=[3, 4, 9]，steps=[0]。
- 按层选择：layers=[5, 14, 27, 3, 8, 17, 4]，heads=all，steps=all。
- 按head选择：layers=all，heads=[4, 3, 9]，steps=all。
- 按step选择：layers=all，heads=all，steps=[0]。

- [搜索方法、选择及正式计时](../intersection_search_v1/REPORT.zh.md)
- [冻结清单及策略文件SHA](protocol.json)
- [完整20组逐样本数据](paired_results.csv)、[汇总](summary.csv)、[逐场景](by_prompt.csv)
- [搜索执行代码](../../../local_experiments/video_runtime_v1/intersection_search.py)
- [验证执行代码](../../../local_experiments/video_runtime_v1/intersection_validation.py)
- [本报告生成代码](../../../local_experiments/video_runtime_v1/report_intersection_validation.py)

视频：每组目录下保存7份video.mp4、latents.pt及result.json；comparison.mp4为两行对比，上排Dense/原交集/搜索均值/搜索均衡，下排按层/按head/按step。

- kitchen_cook：[seed703](videos/kitchen_cook_s703/comparison.mp4)、[seed704](videos/kitchen_cook_s704/comparison.mp4)、[seed705](videos/kitchen_cook_s705/comparison.mp4)、[seed706](videos/kitchen_cook_s706/comparison.mp4)
- forest_run：[seed703](videos/forest_run_s703/comparison.mp4)、[seed704](videos/forest_run_s704/comparison.mp4)、[seed705](videos/forest_run_s705/comparison.mp4)、[seed706](videos/forest_run_s706/comparison.mp4)
- portrait_turn：[seed703](videos/portrait_turn_s703/comparison.mp4)、[seed704](videos/portrait_turn_s704/comparison.mp4)、[seed705](videos/portrait_turn_s705/comparison.mp4)、[seed706](videos/portrait_turn_s706/comparison.mp4)
- reef_fish：[seed703](videos/reef_fish_s703/comparison.mp4)、[seed704](videos/reef_fish_s704/comparison.mp4)、[seed705](videos/reef_fish_s705/comparison.mp4)、[seed706](videos/reef_fish_s706/comparison.mp4)
- tennis_match：[seed703](videos/tennis_match_s703/comparison.mp4)、[seed704](videos/tennis_match_s704/comparison.mp4)、[seed705](videos/tennis_match_s705/comparison.mp4)、[seed706](videos/tennis_match_s706/comparison.mp4)
