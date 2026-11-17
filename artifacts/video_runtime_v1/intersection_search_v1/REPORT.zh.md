# 交集预算的两轮实际视频邻域搜索

固定7层×3head×1noisy step=21个位置，预算21/1392=1.5086%，每个短视频评分10584次。没有增加修正预算或改变权重、候选数、gain阈值。只搜索三个集合的成员。

## 搜索范围与判据

- city_walk、seed701/702、9chunks/105帧、768×512、16FPS、history360、tile72；M16入口、固定编译形状、G2关闭。两seed平均Dense LPIPS为主目标。
- 每轮完整评估154个单层替换（7×22）、27个单head替换（3×9）、3个step替换，共184个邻居。选实际视频平均LPIPS最小者进入下一轮，改进阈值1e-6；最多两轮，重复配置复用结果。
- 除平均最优外，还从所有已测配置选出两个seed各自都不差于原交集（容差1e-6）的平均最优者。这个平衡候选是额外展示，不替代主搜索目标。
- 共测341个不同配置、682份视频，另有初始逐位连续性检查和最后的计时重复。8卡各常驻一个worker，模型、VAE和编译缓存复用；每配置都从chunk0重建历史，未复用其他配置的生成轨迹。
- Dense参考复用既有同prompt/seed的像素缓存；LPIPS(AlexNet)/SSIM在256×384计算，PSNR用原分辨率MSE；指标窗口[21,101)。
- 两个seed同时用于搜索和报告，没有独立验证集；两轮搜索不证明全局最优。若第二轮仍有提升，只能称为有界搜索结束，不能称为邻域收敛。

## 每轮结果

|轮次|进入轮次LPIPS↓|邻域最佳LPIPS↓|本轮改善↑|已测不同配置数|
|---|---:|---:|---:|---:|
|1|0.360136|0.341134|0.019002|185|
|2|0.341134|0.341134|0.000000|341|

## 最终配置

|方案|层|head|noisy step|
|---|---|---|---|
|原交集|[3, 4, 5, 8, 14, 17, 27]|[3, 4, 9]|[0]|
|搜索：平均LPIPS最佳|[3, 4, 8, 14, 17, 25, 27]|[3, 4, 9]|[0]|
|搜索：两seed均不退化最佳|[3, 8, 14, 17, 25, 26, 27]|[3, 4, 9]|[0]|

## 同口径速度与质量

|方案|生成/s↓|相对原交集速度↑|相对完整档速度↑|LPIPS↓|SSIM↑|PSNR/dB↑|
|---|---:|---:|---:|---:|---:|---:|
|C＋Risk基础档|7.6496|1.0102×|1.362×|0.372687|0.382987|14.322|
|完整Full_Pset|10.4187|0.7417×|1.000×|0.363400|0.407230|15.078|
|原交集|7.7280|1.0000×|1.348×|0.360136|0.402107|14.938|
|搜索：平均LPIPS最佳|7.7310|0.9996×|1.348×|0.341134|0.422986|15.396|
|搜索：两seed均不退化最佳|7.7322|0.9995×|1.347×|0.346872|0.422983|15.454|

seed701全部最终方案在GPU6测量，seed702在GPU7；A100-SXM4-40GB、CPU线程2，完整预热一次，3次生成取中位数，再对两seed等权平均。排除模型加载、编译、VAE、IO。搜索worker的单次生成时间只用于吞吐日志，不作正式加速比。基础/完整档质量引用上一轮相同固定形状控制，当前重新计时；原交集和搜索结果重新解码，指标与搜索记录对照。

|方案|seed701 LPIPS↓|对原交集改善↑|seed702 LPIPS↓|对原交集改善↑|
|---|---:|---:|---:|---:|
|原交集|0.328519|+0.000000|0.391753|+0.000000|
|搜索：平均LPIPS最佳|0.285732|+0.042786|0.396535|-0.004782|
|搜索：两seed均不退化最佳|0.305239|+0.023279|0.388504|+0.003249|

![双seed质量分布](search_frontier.png)

虚线标出原交集的两个seed指标；左下区域表示两个seed都改善。所有点来自相同21个位置预算的实际视频。

## 执行与复现

每个worker先对原交集的两个seed生成latent，并与上一轮逐位比较；新鲜解码的控制LPIPS也必须与旧记录相差小于1e-6。每个候选检查10584个评分事件、无非有限分数，固定形状避免已发现的任务顺序影响。所有候选保存latent和逐帧指标；最终候选另保存视频，不在搜索阶段编码数百个重复用途视频。

没有更改全局默认。启用搜索结果：在当前stream命令添加`--full-pset --pset-policy artifacts/video_runtime_v1/intersection_search_v1/presets/best_mean.json`；希望明确使用双seed均不退化判据时改为`best_balanced.json`。文件SHA进入identity，preset强制static_route=True。

实现：[intersection_search.py](../../../local_experiments/video_runtime_v1/intersection_search.py)负责固定预算邻居生成、SQLite领任务、常驻模型、两轮选择和完整排名；[intersection_finalize.py](../../../local_experiments/video_runtime_v1/intersection_finalize.py)负责最终同卡计时、解码和报告。范围执行复用`runtime.py::set_pset_scope`与`pipeline_graph.py`的selected-head路径，没有新增kernel或算法近似。

在仓库根用项目venv执行`python -m local_experiments.video_runtime_v1.intersection_search`。结束后分别执行`intersection_finalize --mode timing --seed 701/702`、`--mode decode --seed 701/702`和`--mode report`。既有结果根保留，重做实验需新根目录，避免覆盖本轮证据。

原始记录：[protocol.json](protocol.json)、[全部排名CSV](ranking.csv)、[结果CSV](comparison.csv)、[两轮及全部配置](summary.json)、[可部署preset](presets/)。每个候选`results/<policy>/s<seed>/result.json`保存逐帧指标；`round1/2_manifest.json`记录邻居，`round1/2_result.json`记录当轮全部分数。

四列并排视频依次为Dense、原交集、平均最优、双seed均不退化最优：[seed701](comparison_s701.mp4)、[seed702](comparison_s702.mp4)。

## 冻结后的跨场景验证

两版均完成5 prompt ×4 seed验证，见[独立验证报告](../intersection_validation_v1/REPORT.zh.md)。两版搜索结果在该验证集的平均LPIPS均不如原交集；本报告搜索集中的改善不能当作跨场景收益，保留原交集低预算方案。
