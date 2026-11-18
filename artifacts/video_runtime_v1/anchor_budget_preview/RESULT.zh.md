# horse_field 首帧锚点预算对照

使用已有 30 秒样本的 prompt embedding、初始噪声及 41 个 chunk 的逐步噪声，seed 2027。输入 embedding 与归档 horse_field 张量逐位相等。两组均采用 base，首个 latent 帧的 1536 个 token（24 个 Page64）永久保留，所有稀疏查询均将其纳入精算。每个查询自己的页也纳入精算；首帧与自页以外的名额按 risk 选择。其他 chunk 的历史页沿用原分组顺序。每次 clean commit 后检查各层、各头的锚点仍包含首帧的全部 token。

| 窗口总页数 | 额外增加首帧 | 固定总预算 |
|---:|---:|---:|
| 216 | 45 | 25 |
| 288 | 52 | 28 |
| 360 | 60 | 36 |
| 432 | 67 | 43 |

额外版精算页数为 24 + floor(P/10)。固定版取 max(25, floor(P/10))，216 页按用户确认升至 25 页。首次两段走现有 dense 路径。两种实验模式均在 preview 入口选择。成片按已有样本导出 480 帧、16 fps，单片 768×512。[生成程序](../../../local_experiments/video_runtime_v1/preview_sink_chunk.py)

新增固定 15% 对照，复用同一输入文件和 seed 2027。通过 `--mode anchor-fixed --fixed-percent 15` 运行，预算包含首帧 24 页和查询自页。四种稀疏窗口实际精算页数依次为 32/216、43/288、54/360、64/432。按窗口页数向下取整，满窗口留给 risk 路由 39 页。
