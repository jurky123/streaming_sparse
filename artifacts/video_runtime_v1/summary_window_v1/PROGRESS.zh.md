# 当前页摘要直接写入分层缓存

基线对 72 个当前页生成 keys、mean、detail、stats，再将四个结果复制到各层对应的缓存切片。12 个头共写 1,230,336 字节 BF16；直接写入省去四次复制，即每次调用约 2.46 MB 的逻辑读写。两个摘要输出 kernel 按层内 head stride 定位结果，继续使用相同的代表键、均值、细节和误差统计公式。clean 阶段沿用原提交逻辑。

| 窗口页数 | 原耗时（ms） | 直接写（ms） | 加速比 |
|---:|---:|---:|---:|
| 216 | 0.1940 | 0.1789 | 1.0844× |
| 288 | 0.1615 | 0.1497 | 1.0787× |
| 360 | 0.1614 | 0.1495 | 1.0790× |
| 432 | 0.1612 | 0.1492 | 1.0800× |

使用捕获的真实 KV、已部署的第 15 层权重和常驻缓存 stride。每种窗口交替测 24 对，各回放 12 次。四个字段均逐位一致，窗口外的哨兵未改变。[模块数据](module.json)

两边均启用 KV 投影直接写缓存。city_walk/701 三档各交替运行 10 对；其中两档有个别计时回退，只对这两档在 kitchen_cook/703 和 tennis_match/703 各补测 6 对。每行均在同一张卡上比较新旧实现。

| 视频 | 档位 | 旧版（s） | 直接写（s） | 加速比 | 成对提速 | 每对节省中位数（ms） |
|---|---|---:|---:|---:|---:|---:|
| city_walk | [base](base.json) | 5.19870 | 5.18792 | 1.0021× | 10/10 | 12.5 |
| city_walk | [intersection](intersection.json) | 5.32860 | 5.31359 | 1.0028× | 9/10 | 12.5 |
| city_walk | [step](step.json) | 5.99901 | 5.99053 | 1.0014× | 8/10 | 10.4 |
| kitchen_cook | [intersection](repeat_kitchen_cook_703/intersection.json) | 5.29316 | 5.28238 | 1.0020× | 6/6 | 13.0 |
| kitchen_cook | [step](repeat_kitchen_cook_703/step.json) | 6.00981 | 5.99975 | 1.0017× | 6/6 | 10.7 |
| tennis_match | [intersection](repeat_tennis_match_703/intersection.json) | 5.33357 | 5.32481 | 1.0016× | 6/6 | 12.6 |
| tennis_match | [step](repeat_tennis_match_703/step.json) | 6.02558 | 6.02385 | 1.0003× | 6/6 | 12.9 |

54 对中 51 对更快。三条视频的全部对照均得到逐字节相同的最终 latent 与一致的路由计数。复现使用 `python -m local_experiments.video_runtime_v1.check_summary_window` 和 `python -m local_experiments.video_runtime_v1.check_rms_row_video --tier base --summary-window --pairs 10`。

[KV 投影验收](../cached_projection_v1/PROGRESS.zh.md) · [融合注意力寄存器实验](../fused_lifetime_v1/PROGRESS.zh.md)

GPU7 原生 Dense 更新为 6.56527 秒。两项优化均启用后，默认入口 5 次中位数 base 5.25095 秒、intersection 5.32040 秒、step 5.95632 秒，三档 latent 与上次部署逐字节相同。base 相对同卡 Dense 提速 1.25030×，距 4.68834 秒目标还差 0.56261 秒。 GPU 时间线中 DtoD 复制累计从 91.69 降到 25.97 ms；融合注意力约 856 ms。[独立入口](deployment.json) · [时间线](device_timeline.json)
