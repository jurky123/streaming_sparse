# 消融方案（2026-11-03）

保持512×768、338历史页（layer0为360）、16 probes、精确页预算10%、原始输入和seed。层0、前两个chunk和clean仍Dense。所有候选冻结权重，不对移除模块的版本补训；保留原有摘要分配，不利用省出的空间扩大KV。只验质量，不作共享GPU速度结论。

## 第一阶段：同Q/K/V的局部因果对照

horse_field、traffic_crossing，seed101；chunk2/5/8，step0/2；层1/8/15/22/28/29；两个query tile各16 probes、12heads。每场景36事件，每事件10组：

|组|改变项|控制条件|
|---|---|---|
|full|完整最终方案|参考|
|projected_mean|完整V均值改为当前basis和旧center的rank8重建|K、detail、mask完全相同|
|no_detail|Haar细节置零|K、完整均值、mask完全相同|
|no_tail|不补偿未选页，仅精确选中页并独立归一化|同一精确mask与raw KV|
|mass_route|按代表softmax页质量选页|同一最终摘要、页数、own页保护|
|old_route|使用旧方案在该状态的原始选页结果|同raw KV及最终补偿器；旧路由内部使用旧摘要，仅作补充黑盒对照|
|initial_fixed_mask|新结构、继承旧权重、不做本轮训练|固定最终方案mask和物理分组|
|pretrained_fixed_mask|仅摘要预训练，不做两步下游训练|固定最终方案mask和物理分组|
|initial_own_route|未训练新结构及其自身新路由|相同raw KV、物理分组和预算|
|pretrained_own_route|摘要预训练及其自身新路由|相同raw KV、物理分组和预算|

局部实验不重新分组，不能据此判断重排收益；路由的严格同摘要对照是mass_route。输出逐事件NMSE、平方误差/目标能量、log分母误差和mask差异；汇总同时报告事件平均与能量加权NMSE。相关事件不用于虚构独立显著性。

## 第二阶段：前段闭环视频

两场景seed101，9chunk/105帧，主窗口RGB[21,101)。复用此前Dense、旧版、最终新版同输入结果，新增6组：projected_mean、no_detail、mass_route、no_grouping、no_new_training、no_downstream。

均值/细节消融在路由完成后作用于输出补偿分支；路由仍读取完整统计。它测量补偿分支的贡献，不冒称单独优化过的替代系统。闭环中不同输出会自然引起后续query/mask变化，因此“相同mask”的严格因果解释只适用于第一阶段。

no_grouping不做响应重排，保留自然连续页和相同partial-chunk淘汰页次序；K/V配对、RoPE和持久空间配置不变。no_new_training与no_downstream的分组参数也分别恢复到对应checkpoint，衡量整个训练阶段的总贡献。

## 实施与验收

代码独立于正在运行的正式验收，未修改其源文件或checkpoint。CPU检查已通过：数据不被原地修改、秩8投影、mass归一化、关闭分组时K/V配对与排列双射。GPU首项必须验证full控制分支在前三个chunk与既有最终方案latent逐位一致，通过后才开始消融。

消融已经加入串行计划；当前视频先完成，之后优先执行消融，原有完整视频验收任务仍保留。所有任务独立进程，局部逐事件保存，视频按clean latent恢复。不存大QKV。结果在artifacts/review_v1_20261102/ablation/，Markdown汇总为ablation_results.md。没有结果的组不填推测值。
