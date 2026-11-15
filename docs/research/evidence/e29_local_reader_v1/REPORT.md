# E29：页面局部响应在线Reader验证

**结论：局部响应目标改善了部分在线决策能力，但两个种子未同时通过可靠性门槛，Test继续封存。真实beta替换显著提高收益与条件相关，质量修正及其与响应误差的耦合是更明确的诊断线索；尚不能归因为唯一原因。**

固定E24 C摘要、E27 Adaptive phi、B16、Risk、Page64/R4、10%/own、M16与Medium一次交换。没有采集QKV或扩大网络。两种子分别4608次更新，开发集两轨迹288事件；按预登记门槛决定是否解封Test。

## A：历史/当前教师依赖

|教师配置（Set）|seed17 Oracle恢复|seed29 Oracle恢复|
|---|---:|---:|
|FullTeacher|45.52%|48.41%|
|HistoryOnly|10.84%|18.61%|
|CurrentOnly|-0.49%|2.87%|

FullTeacher完整重现E28 Local-LOTO。缺失修正的页面同时令z=0、beta=0（u=0），其分母仍为原摘要质量，未保留真实分母。历史教师为跨事件LOTO、当前教师为本事件Oracle，全部不可部署。两类信息对集合目标有非线性共同作用，不能将两单支收益相加解释。

## B/C：实际收益与教师替换

|Set信息|seed17执行恢复|seed29执行恢复|seed17提议+真实门控|seed29提议+真实门控|
|---|---:|---:|---:|---:|
|E27-B|10.67%|6.85%|19.31%|14.86%|
|Local-Online|11.59%|24.62%|25.67%|30.97%|
|Local-TrueBeta|46.37%|55.48%|55.45%|59.19%|
|Local-TrueResponse|30.33%|33.08%|34.28%|36.60%|
|E28-LocalLOTO|45.52%|48.41%|48.39%|53.33%|

恢复率为Σ真实执行收益/Σ同池Q64最佳正收益，所有方法摘要/初选和分母一致。Local-Online只有摘要输入，是本轮主结果；TrueBeta/TrueResponse及LOTO使用教师信息，仅诊断。事后真实门控也不可部署。

|seed|决策|Risk NMSE总和|决策后NMSE总和|Oracle后NMSE总和|汇总改善|逐例平均改善|退化/no-op|P10|最差|
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
|17|Set|9.405315|9.254260|8.101717|+1.606%|-63.394%|95/9|-7.942%|-16910.351%|
|17|Independent|9.405315|9.345056|8.101717|+0.641%|-63.815%|107/9|-6.604%|-16910.351%|
|29|Set|10.019240|9.703450|8.736607|+3.152%|+0.471%|91/7|-7.348%|-65.814%|
|29|Independent|10.019240|9.898577|8.736607|+1.204%|+0.678%|106/6|-5.138%|-37.919%|

正收益、近零已执行、逐例相对指标有效样本数见CSV；近零初始NMSE沿用1e−12阈值，仅影响相对指标，不删除汇总样本。No-op就是Risk，修正收益0。

## 集合条件变化

|seed|信息|双重中心化相关|逐事件交互相关中位数|收益符号翻转召回|
|---|---|---:|---:|---:|
|17|Local-Online|0.1291|0.4273|9.60%|
|17|Local-TrueBeta|0.5121|0.6012|12.34%|
|17|Local-TrueResponse|0.0986|0.5708|12.27%|
|29|Local-Online|0.1063|0.4262|8.92%|
|29|Local-TrueBeta|0.6659|0.6014|10.78%|
|29|Local-TrueResponse|0.1279|0.5777|11.31%|

沿用E25五个状态、共同合法动作面板，固定页面预测，只更新集合合成；没有重拟合教师来迎合状态。原尺度汇总相关对极端幅度敏感，逐事件中位数与翻转召回同时保留。相对退化的尾部可能由近乎精确的初始状态放大；tail_cases.csv同时记录初始/最终绝对NMSE和收益，不裁剪样本，也不以极端百分比替代绝对误差。

## 训练公平性与变量

网络仍为712→64 GELU→85，共51157参数；BF16页面系数、输出16维局部响应与质量系数。C、phi逐tensor冻结；初始权重与E27-B相同，训练轨迹/顺序/4608更新/Adam设置/60-20-20集合状态规则相同。每epoch评价，只有Local-Online Set汇总真实收益用于选检查点；教师替换不参与选择。

按本轮指定，将原中心化Huber换为mass加权局部平方误差，除以训练集固定median RMS²作单位归一化；没有使用逐页响应尺度消掉mass权重。beta Huber及集合gain/listwise/no-op沿用E27-B归一化。此比较同时包含目标、解析重构和指定局部损失形式的改变，不能仅归因于目标命名；没有开展损失权重搜索。

页面编码器只读现有摘要；phi冻结。全局中心由摘要计算，真实页面输出/质量只生成训练标签或离线指标。预测路径不读取原始KV、Dense或教师残差。数学组合使用稳定logmass与expm1，零修正恢复原摘要，不裁剪beta或交换分母。

## 开发门槛

- seed17：通过=False；{"positive_gain_each_trajectory": true, "set_better_independent": true, "oracle_recovery_ge20pct": false, "nonnegative_mean": false, "degradation_le15pct": false, "p10_better_than_E27": false, "worst_better_than_E27": false, "conditional_corr_better": false, "signflip_recall_better": true}；轨迹收益={'kitchen_cook_s701': 0.07122737990532571, 'snow_ski_s702': 0.07982751409090173}
- seed29：通过=False；{"positive_gain_each_trajectory": true, "set_better_independent": true, "oracle_recovery_ge20pct": true, "nonnegative_mean": true, "degradation_le15pct": false, "p10_better_than_E27": false, "worst_better_than_E27": true, "conditional_corr_better": true, "signflip_recall_better": true}；轨迹收益={'kitchen_cook_s701': 0.2321988009453534, 'snow_ski_s702': 0.08359152552598918}

可靠性中的“显著降低”在运行前量化为退化≤15%、P10和最差均优于E27-B；要求相关和翻转召回也均改善。未根据结果放宽。门槛总判定：未通过，Test继续封存；不追加训练、扩大模型或调参。

两个种子的Set汇总收益均优于同信息Independent，且两条开发轨迹汇总收益均为正；但Seed17恢复不足20%、逐例均值和尾部仍失败，两种子的退化比例均超过门槛。事后真实门控仅恢复25.67%/30.97%，仍低于真实beta干预，不能只靠加接受门控解释全部缺口。

局部响应监督和解析中心路径确有进展，但真beta是教师信息，不能把46.37%/55.48%当作在线结果。两种教师替换改变了误差抵消，不能据此证明beta是唯一瓶颈或R4缺乏信息。按计划停止追加训练，不更改学习率、层数、特征数或损失权重。

尚未进行Kernel、端到端视频或实际延迟验证。


## 数值审计与成本

{
  "action_checks": 3456,
  "all_count_own_valid": true,
  "summary_only_exact_actions": 12,
  "summary_only_checked": 12,
  "max_summary_only_score_difference": 9.450773497121645e-15,
  "max_independent_predicted_gain_error": 6.938893903907228e-18,
  "max_true_gain_error": 1.0361358541988164e-09,
  "frozen_C_phi": true,
  "training_updates": [
    4608,
    4608
  ]
}

独立审计从Query与摘要重新计算logmass和中心，不接受原始KV、教师质量或Dense作为预测输入；动作与评价路径一致。longdouble混合输出独立复核预测分数及真实动作收益，另对全部主/教师替换动作检查预算、own和合法面板。离线Dense能量仅用于NMSE单位，不改变选页或no-op。

Reader仍为51157参数，页面派生系数170B＋冻结phi64B，总计234B/page；并未满足旧192B目标。新的解析路径需要投影摘要V和组合中心：可选缓存四个16维投影代表128B/page（历史29层约14.36MiB），或重算/复用；预计新增低维R4 AV约0.726GFLOPs/层，组合约0.453GFLOPs/层，另有commit投影、指数归一化和IO。原始KV探测为0，但不声称总在线成本完全不变，也未测融合延迟。
