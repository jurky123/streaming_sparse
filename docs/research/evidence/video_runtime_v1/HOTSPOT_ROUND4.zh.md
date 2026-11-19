# 第四轮：FP64 动作 kernel 与 Dense／视频误差对照

## profile 指向的优化

上一轮 profile 的最大单 kernel 是 FP64 交换动作的融合归约，约占路由 CUDA kernel 时间 39%。虽然动作仅 16×32、M16、B16，通用融合器对每个动作仍计算完整五组回退向量，并沿输出坐标重复昂贵的 exp/expm1 标量处理。

新增 `scores_action.py`：每个 CTA 负责一个交换，16 个 Query 上的 FP64 质量系数先计算再广播到 16 个输出坐标；主导页／非有限端点检测满足时才执行五组正质量回退。保持 FP64 libdevice exp/expm1、原端点增量收益及严格 >0 no-op，不改预算、候选规则或训练目标。禁用 FMA contraction 保持表达式顺序。通过 custom_op 接入整路由编译与 CUDA Graph。开启方式 `VideoRuntime(..., compile_route=True, compile_tail=True, action_kernel=True)`；默认保留原路径。

## 模块和短视频验收

同一 city_walk 捕获事件扩展的 72 head–tile 批，CUDA Graph 25 次中位 GPU 时间：已编译的旧评分器 1.0844 ms → 新动作 kernel 0.44544 ms，2.434×。最大评分误差 5.55e-17，argmax/no-op 一致。14 个 E49 数值案例和主导页相差 1000 log 单位的回退测试通过。该 kernel 的主算术量仍为 72×16×512×16 个坐标的端点/增量计算；收益来自系数复用与避免常规动作的回退计算，不是改变精算预算。

此前 9 chunks／338 页短视频已完成验收：37.8187 → 32.5664 秒，1.161×；路由 profile 从21.961秒降到16.699秒。Attention约8.25秒保持。**新动作 kernel 相对上一轮融合补偿视频的 latent 逐位相同（NMSE=0）**，这轮未增加该视频的数值漂移。

测量脚本、数值门禁、原始计时和 latent 对照保存在 `artifacts/video_runtime_v1/hotspot_round4/`。

## 正式测试配置修正

仓库 `docs/streaming_evaluation_20261102/report.zh.md` 正式配置为 41 chunks、480帧/16FPS、512×768、最多5个历史chunks；早期报告明确稀疏338页而Dense360页。后续 `stream_eval_latest_*` 脚本已统一 `Config(history_pages=360)`，并声明“equal history，不声称 equal memory”。此前热点测试的9chunks/338页是短视频研究配置，不能代替正式配置的速度结论。

按用户最新要求，正式配置固定为短视频9chunks／105帧与15秒21chunks／240帧；历史均360页、当前72页，共432页。30秒任务已取消。使用同一个city_walk/seed701和配套新全覆盖C＋Full_Pset seed17。已完成正式测量见 `FORMAL_SHORT_15.zh.md`；理论计算、优化前后误差与本轮结项结论见 `ROUND_CLOSEOUT.zh.md`。15秒既有数据保留在 `artifacts/video_runtime_v1/formal_15_30/15s/`，新短视频在 `formal_short_15/short/`。
