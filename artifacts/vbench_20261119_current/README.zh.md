# VBench 停机快照

2026-11-19 UTC。五档共 5,760 条 MP4 均已生成并核对路径和非零大小。intersection 使用原始权重、10% 非锚点条件预算及首帧时间 RoPE 重定位。

评分队列已登记 4,728 / 20,890 项，均属短视频。官方逐视频结果实际返回 4,697 项，color 有 31 条没有逐视频值。下表仅对有返回值的视频取均值。长视频八个维度尚无评分。

## 生成耗时

每档短视频 768 条、长视频 384 条。单位为秒，各列为逐视频中位数，加速比以同长度 Dense 生成中位数为分子。

| 时长 | 方法 | 生成 | VAE 解码 | 生成加速比 |
|---|---|---:|---:|---:|
| short | dense | 4.651 | 3.285 | 1.000× |
| short | lightforcing | 4.377 | 3.304 | 1.063× |
| short | base | 3.818 | 3.291 | 1.218× |
| short | intersection | 4.217 | 3.309 | 1.103× |
| short | step | 4.388 | 3.316 | 1.060× |
| long | dense | 35.398 | 19.991 | 1.000× |
| long | lightforcing | 28.372 | 19.940 | 1.248× |
| long | base | 25.679 | 19.940 | 1.378× |
| long | intersection | 29.413 | 20.043 | 1.203× |
| long | step | 30.323 | 19.989 | 1.167× |

## 短视频维度均值

imaging_quality 逐视频值除以 100。各维度保留自身量纲。覆盖列为有逐视频值的条数 / 计划条数。

| 维度 | 覆盖 | Dense | LightForcing | base | intersection | step |
|---|---:|---:|---:|---:|---:|---:|
| aesthetic_quality | 330 / 330 | 0.6687 | 0.6628 | 0.6624 | 0.6643 | 0.6636 |
| appearance_style | 360 / 360 | 0.1937 | 0.1927 | 0.1920 | 0.1922 | 0.1922 |
| background_consistency | 384 / 390 | 0.9632 | 0.9603 | 0.9597 | 0.9639 | 0.9591 |
| color | 289 / 320 | 0.8879 | 0.8919 | 0.9035 | 0.8966 | 0.8887 |
| dynamic_degree | 320 / 320 | 0.7188 | 0.7344 | 0.7344 | 0.7031 | 0.7188 |
| human_action | 370 / 370 | 0.8378 | 0.8919 | 0.8514 | 0.8784 | 0.8378 |
| imaging_quality | 330 / 330 | 0.6993 | 0.6976 | 0.7003 | 0.7036 | 0.7029 |
| motion_smoothness | 320 / 320 | 0.9848 | 0.9825 | 0.9843 | 0.9845 | 0.9842 |
| multiple_objects | 350 / 350 | 0.7277 | 0.7063 | 0.7259 | 0.7179 | 0.7071 |
| object_class | 290 / 290 | 0.7597 | 0.7554 | 0.7780 | 0.7780 | 0.7769 |
| overall_consistency | 330 / 330 | 0.2495 | 0.2500 | 0.2507 | 0.2501 | 0.2503 |
| scene | 384 / 390 | 0.2612 | 0.2776 | 0.2638 | 0.2516 | 0.2574 |
| spatial_relationship | 0 / 400 | — | — | — | — | — |
| subject_consistency | 320 / 320 | 0.9560 | 0.9519 | 0.9520 | 0.9550 | 0.9486 |
| temporal_flickering | 320 / 320 | 0.9877 | 0.9836 | 0.9884 | 0.9891 | 0.9876 |
| temporal_style | 0 / 390 | — | — | — | — | — |

jobs.sqlite 保存任务与评分进度，timings.csv、scores.csv 和 dimension_summary.csv 提供逐条计时、逐视频评分及覆盖率。scores.tar.gz 包含登记批次的原始任务清单与结果。新机器按 reproduction/vbench_current/README.zh.md 准备环境与队列，恢复 videos/ 和此 jobs.sqlite 后启动 launch.py 即可续评。

本轮全部 MP4 已转存至 LeoFS 共享盘 /home/shuchenweng/jzk/vbench_archives/20261119_current/videos.tar（31,709,798,400 字节）。videos.csv 记录归档内逐视频路径与字节数，已与源文件大小核对。videos.tar.sha256 可供迁移后校验。共享盘同目录的 run-records.tar.gz 保留运行日志。
