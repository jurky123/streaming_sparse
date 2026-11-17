# 审查意见落实：worker统一、Reader存储压缩与分母端点复用

## 结论与范围

本轮完成P0和三项P1，并作为新部署基线。稳定满窗口路由 4.620→4.304ms（1.074×）；9-chunk生成 10.772→10.522s（1.024×），含VAE 15.071→14.839s（1.016×）。latent及全部RGB帧逐位一致。没有拿回此前表中大部分参考差额；实际收益主要来自分母减少重复非线性计算。

保留Page64/Q64/R4、完整128维V均值＋rank8、M16/B16、10%精算、16×32交换＋no-op，全覆盖C_s17和Full_Pset_s17，旧快精算/补偿、RoPE、A-only/split Reader、scan、projected-risk和cached-logmass。PAGE32/lazy关闭。没有训练、没有更换权重、没有新精度近似。本轮未展开P2小WORK流水和P3跨Query复用，也未启动大批量质量评估。

## P0：批量worker已与验收配置统一

新增 [deployment.py](../../../local_experiments/video_runtime_v1/deployment.py)，`CURRENT/current`定义当前部署，`configure`设置路由开关，`load_weights`加载全覆盖摘要和Reader，`make_runtime`统一创建运行时。benchmark与worker调用同一组函数。benchmark保留实验控制开关，当前完整命令见 [CURRENT_VIDEO_BASELINE.txt](../../../local_setup/CURRENT_VIDEO_BASELINE.txt)。

[worker.py](../../../local_experiments/video_runtime_v1/worker.py) 默认摘要C_s17、Reader Full_Pset_s17，history360、tile72、legacy rope/attention和全部已验收开关。metadata从实际配置生成，历史页数不再写死338；新增Reader checkpoint、history、local-assets参数，去掉placeholder状态标签。运行时源码身份涵盖新增存储/分母模块。

worker实测：同一个GPU4进程连续完成Dense和c_pset两个9-chunk任务，只加载一次模型；c_pset latent与上轮已验收benchmark逐位一致。另一个postprocess进程完成两任务解码、视频编码及指标，Dense像素参考只生成一次并复用。队列为2个生成done＋2个后处理done。此验证在三项P1提升为默认前完成；最终worker默认配置与本轮已验收benchmark逐字段核对一致，权重路径归一化后比较。证据 [worker_verified](../storage_review_v1/worker_verified/)、[worker_config_equivalence.json](worker_config_equivalence.json)。worker时间含冷编译/latent IO，不能作稳态速度比较。

## 三项P1的实现与精度边界

|改动|实现位置与关键入口|实际变化|
|---|---|---|
|Reader输出FP32存储|[reader_reduce.py](../../../local_experiments/video_runtime_v1/reader_reduce.py)：`reader_reduce`、fake注册、`compact_scores`|state/logs/cand使用显式`output_float`参数决定容器dtype；后续logs softmax、端点加法、向量聚合在读取时提升FP64。|
|FP64消费者与fallback|[scores_moments.py](../../../local_experiments/video_runtime_v1/scores_moments.py)：`moments`；[scores_packed.py](../../../local_experiments/video_runtime_v1/scores_packed.py)：`packed_kernel`|分字段加载后提升FP64再运算；fallback也在`lr+br`、`ca+pa`之前提升。不重新物化整块FP64 CAND。|
|CO/A BF16持久缓存|[page.py](../../../local_experiments/video_runtime_v1/page.py)：`encode`；[runtime.py](../../../local_experiments/video_runtime_v1/runtime.py)：编码缓存分配/`encode`；[reader_split.py](../../../local_experiments/video_runtime_v1/reader_split.py)：`feature`、两个kernel|保留原来已舍入的BF16数值；持久缓存、Graph缓存保持BF16，load后显式转FP32，乘加次序保持。mean/detail不改，不是PAGE32。|
|分母端点组合|[endpoint_den.py](../../../local_experiments/video_runtime_v1/endpoint_den.py)：`denominator`、`cached_add`、`add_kernel`；[reader_reduce.py](../../../local_experiments/video_runtime_v1/reader_reduce.py)：`ENDPOINT_DEN`分支|先组成移出侧U_r及加入侧V_a，再展开16×32。V_a设custom-op边界，防止编译器重复计算。保持FP64正质量及原fallback。|

### 数据量与已实现工作量

当前形状H12/T72/M16/P432。Reader三个输出从203.489MB变为101.745MB：CAND180.486→90.243MB，STATE21.234→10.617MB，LOGSTATE1.769→0.885MB；WORK仍为283.116MB。本轮没有修改WORK生命周期。按输出各写/读一次计，共减少203.489MB逻辑流量；不把这个数字当实测HBM流量。

CO/A单层缓存2.426→1.213MB。按72×16次重复消费计算，逻辑读取2.795→1.398GB；向量响应系数约1.911→0.956GB。这些静态系数可命中L2，不能直接除以HBM带宽预测省下多少时间。直接数据契约测试证明CO/A转回FP32和三个Reader输出均逐位相等。

分母正质量：`U_r=B+R_-r+Zhat_r`、`V_a=A_-a+Z_a`，最终`logaddexp(log U_r, log V_a)`。原每probe 1552次，目标576次。第一次只改torch公式时，编译器把V_a重新内联到512动作里，实际为1056次（仅减少31.96%）；该版本路由约1.055×，短视频也逐位通过。发现后加入V_a显式预计算边界，最终实际576次，21,454,848→7,962,624次、减少62.89%。移出侧2×221184项、加入custom-op442368项、最终7077888项。编译证据 [compiler_evidence](../storage_review_v1/compiler_evidence/)。

没有引入`当前总质量−主导页质量`，没有改变gain>0规则；分母结合顺序仍会改变FP64末位。本轮最初Reader输出dtype使用模块全局变量导致fake元信息缓存不匹配并出现非法访问，已修正为custom-op显式布尔参数，真实分配与fake返回一致。失败运行单独留日志，最终测试重新启动完成。

## 同输入完整路由对照

每项在独占GPU上做before/after配对、交替预热，各50次CUDA Graph稳态计时。输入使用旧快Attention轨迹的真实捕获，切片为216/288/360/432页。数据编码/权重固定。GPU0–7分配独立任务，无同卡并发计时。

|改动|216页加速|288页加速|360页加速|432页加速|
|---|---:|---:|---:|---:|
|仅输出FP32|1.034×|1.046×|1.029×|1.021×|
|仅CO/A BF16|1.016×|1.040×|1.034×|1.028×|
|仅分母重组，未阻止内联|1.007×|1.047×|1.041×|1.034×|
|最终三项组合，显式端点缓存|1.056×|1.114×|1.090×|1.074×|

432页单独显式端点缓存加速1.043×；最终组合两次独立满窗口复验约1.072×与1.074×，不是把单项倍数相乘。所有窗口最终页LUT、mask和统计一致。完整路由返回的Risk数组在部分编译配置下有小差（最大0.15625，最终复验0.001953125），未逐指令定位此非目标模块差异；因此不把“路由输出全部逐位一致”作为结论。相同固定Risk/Reader输入的契约检查另给全部512评分、动作和no-op结果。

## 稳定profile与修正后的理论距离

下表来自最终组合的同输入、预热后profile。Reader按scalar+vector相加，不再重复计custom-op；同名CPU包装与GPU事件不重复计数。Attention未改，用此前相同实现隔离值。模块不构成完整无遗漏总和。

|模块|本轮前/ms|本轮后/ms|参考下限/ms（后）|实测/参考|优化空间|
|---|---:|---:|---:|---:|---|
|旧快精算＋补偿|1.295|1.295|0.771|1.68×|非本轮热点；沿用相同隔离值|
|Reader标量＋向量|1.971|1.912|0.527|3.63×|仍是首要热点，P2小WORK仅列后续原型|
|Reader log质量|0.219|0.219|0.189|1.16×|本轮未改，主要逻辑payload已接近参考|
|Risk主要路径|0.465|0.471|0.377|1.25×|本轮未改，包含概率和原FP32投影|
|分母（含背景log质量、端点及动作组合）|0.331|0.117|0.045|2.61×|参考只计最终输出；不能用于推断非线性峰值|
|512动作标量评分|0.336|0.337|0.089|3.76×|分母已减少重复计算，评分kernel本轮不改|
|排除候选正质量扫描|0.291|0.244|0.143|1.71×|输入组织随紧凑输出变化，算法不改|

Reader子阶段：scalar 0.485→0.542ms（反而变慢），vector 1.486→1.371ms；合计只省约0.058ms。不能把输出字节减半解释成Reader快一倍。分母整体从0.331→0.117ms，才是本轮最明显的局部收益。

理论口径采用本机复制1266.396GB/s、BF16大GEMM202.135TFLOP/s。Reader原0.447ms只计WORK写读566.231MB；加原输出写203.489MB后为0.608ms，当前输出写101.745MB则为0.527ms，仍未计Z/LM输入、系数加载、非线性和归约。这个下限随实现变化，不能把实测减去下限当作可收回的预算。输出消费读取发生在后续模块，不在Reader这行重复算。

Risk参考包含Z/prob和U逻辑payload477.757MB。动作评分参考仅主要den/dot输入113.246MB；分母参考仅最终输出56.623MB，未计额外端点中间量与exp/log1p。FP64大GEMM吞吐不用于推断非线性或索引/扫描延迟；分母的改善使用实际调用量和完整profile说明。扫描参考仍仅唯一输入/输出180.486MB，实际存在归约通信。

## 短视频、误差及worker验收

最终视频GPU3同卡顺序，city_walk seed701，9 chunks/105帧、768×512、16FPS、5个历史chunks/360历史页。完整长度预热后2次生成，VAE单独预热；不含模型加载/编译/视频编码/文件IO。首次纯公式版本验收后因编译证据发现未完全共享，又针对最终explicit-cache版本做了第二组配对，前一组保存于 [storage_video_v1](../storage_video_v1/)，没有覆盖或择优删除。

|阶段|前/s|后/s|加速|
|---|---:|---:|---:|
|生成|10.772|10.522|1.024×|
|VAE|4.299|4.317|0.996×|
|合计|15.071|14.839|1.016×|
|route_and_correction|3.496|3.250|1.076×|
|attention|1.852|1.851|1.001×|
|summary_and_encoding|0.442|0.439|1.008×|
|clean_commit|0.875|0.876|1.000×|

阶段计时来自单独插桩运行，不与墙钟简单相加；摘要/编码、clean commit及VAE尚无完整工作量审计，不填虚构理论倍数。

|优化误差指标|最终结果|
|---|---:|
|latent NMSE|0|
|全部RGB MAE /255、最大绝对差|0、0|
|PSNR|100.0dB（零差封顶）|
|SSIM|0.999999572|
|LPIPS|0.0|

PSNR/SSIM/LPIPS窗口[21,101)，全部帧RGB另作逐位检查。SSIM末位偏差来自指标实现。两组视频latent均与上轮基线连续对应；本轮前latent确认等于已接受版本。没有用相对Dense算法误差作为优化验收。

固定输入中三个Reader输出还原FP64后逐位一致，输出32的后续评分最大差2.08e-17，最终组合最大2.78e-17，全部动作/no-op一致。14份E49包含已有微小正gain；主导页1000-log差和强制fallback重建均通过。FP32容器在普通/强制fallback两路径与FP64容器结果逐位相同。证据 [contracts_cached.json](../storage_review_v1/contracts_cached.json)、[endpoint_cached_fixtures.json](../storage_review_v1/endpoint_cached_fixtures.json)、[errors.json](errors.json)、[quality.json](quality.json)。

## 复现与后续边界

新增开关 `--reader-output32 --coeff-bf16 --endpoint-den --cached-endpoints`。`deployment.CURRENT`已启用四项，worker直接使用；benchmark实验默认仍保留显式开关以复现对照。需要新输出目录/新队列，旧队列identity不可混用。

- 完整路由：[profile_storage_route.py](../../../local_experiments/video_runtime_v1/profile_storage_route.py)，`--variant output32/coeff/den/den_cached/combined_cached --pages 432`，最终profile加`--steady-profile --profile-json`。
- 数据契约：[check_storage_contract.py](../../../local_experiments/video_runtime_v1/check_storage_contract.py)。
- E49/fallback：[check_endpoint_fixtures.py](../../../local_experiments/video_runtime_v1/check_endpoint_fixtures.py)，最终路径加`--cached`。
- 短视频：[run_storage_cached_video.py](../../../local_experiments/video_runtime_v1/run_storage_cached_video.py)，使用脚本OUT新目录；指标 [quality_storage_cached_video.py](../../../local_experiments/video_runtime_v1/quality_storage_cached_video.py)。
- 报告：[report_storage_review.py](../../../local_experiments/video_runtime_v1/report_storage_review.py)，只读已保存证据，不重新跑GPU。

后续可以按审查建议先试WORK 512/1024行生产消费，再考虑vector-only跨Query复用；本轮不重复PAGE32/lazy/寄存器参数筛查。当前结果支持结构性优化有收益，但不支持简单拿回大部分理论差额。
