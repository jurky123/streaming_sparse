# 残差与归一化融合

旧版 base 时间线中，逐元素残差核累计 72.5 ms，向量化 LayerNorm 累计 44.4 ms。一次处理 4608×1536 个 BF16 元素，残差张量为 14.16 MB。两处融合各省去一次完整残差读取，归一化均沿用 D1536 的均值和方差归约顺序。第一处保留门控乘法和残差相加的 BF16 舍入，再做带权重和偏置的 LayerNorm；第二处将 cross-attention 残差与后续 LayerNorm、帧调制一起计算。两处同时输出残差与归一化结果。

模块用 0.01、1、16 三种输入尺度分别测 30 对，每对每侧回放 40 次，两种输出逐字节一致。
门控残差与仿射归一化加速 1.333–1.414×。[模块样本](gated.json)
加法残差与帧调制归一化加速 1.127–1.160×。[模块样本](module.json)

同一卡上交替运行旧版和融合版，每档使用 city_walk/701、105 帧。四种配置共 40 对均更快，最终 latent 与路由计数一致。

| 视频 | 档位 | 对数 | 旧 / 新中位数（s） | 加速比 | 更快次数 | 成对节省中位数（ms） |
|---|---|---:|---:|---:|---:|---:|
| city_walk | [base](base.json) | 10 | 5.18686 / 5.15698 | 1.00579× | 10 | 30.99 |
| city_walk | [intersection](intersection.json) | 10 | 5.31186 / 5.28084 | 1.00587× | 10 | 30.89 |
| city_walk | [step](step.json) | 10 | 5.96978 / 5.94008 | 1.00500× | 10 | 28.71 |
| city_walk | [dense](dense.json) | 10 | 6.61044 / 6.58605 | 1.00370× | 10 | 22.84 |

独立入口 GPU7 原生 Dense 6.54073 秒、base 5.19029 秒，均测 5 次。GPU 时间线上独立逐元素核降至 35.4 ms，原向量化 LayerNorm 降至 1.2 ms；相应计算进入两个融合核。[部署结果](deployment.json) · [时间线](device_timeline.json)

模块复现使用 `python -m local_experiments.video_runtime_v1.check_residual_norm`，门控路径加 `--gated`。视频配对用 `check_rms_row_video --residual-norm`。
