Streaming KV v0.5 — Per-query Signed Block Interaction Routing

数据与隔离
- 复用../v04/data/full真实QKV，全部12物理heads；所有v0.5输出在本目录。
- 原v0.3/v0.4源文件、协议、训练、评估和评审包保持不变。
- configs/protocol.json记录完整配置、数据划分和实现选择。
- reports/progress.html与PROGRESS.txt每15秒显示实际阶段、步数、Val和评估进度。

模型
- 历史F与原Euclidean soft-kmeans/repair/STE/Mean-Max-Min算法不变；增加PairNorm与1层可缓存Pre-Q Transformer。
- 每个原始Query独立经过128→256→128 projection、PairNorm和log(norm(q))标量投影。
- 残差Fusion：[LN(h),e,LN(h)*e]→384→256→128，scale初始0.1，输出PairNorm。
- R1：普通2层Interaction+LinearScore。
- R2：普通2层Interaction+Relevance/Correction。
- R3：2层Signed Interaction+同结构Relevance/Correction。
- Signed u为128→128→64+PairNorm；有效Blocks去中心、L2归一化，FP32 Gram关系。
- 排除Self/Empty；按sum(abs(R))+1e-6归一化消息，消息投影无有效bias，残差scale初始0.1；FFN512。
- Correction是相同MLP对有消息/零消息输入的差，保证零交互消息时精确为0；lambda初始0.1可学习。
- 原始K/V、Attention Logits及Statistics不做LayerNorm。
- 两组共享历史F/Partition/Aggregation/Pre-Q H，也严格共享一个可训练Query Projection/Fusion。
- 每tile的e_q和x_q,b只计算一次，全部分支直接使用同一个Tensor；独立评分/交互模块的可兼容初始权重一致。
- R1/R2/R3单模型参数：1,440,332 / 1,531,926 / 1,538,710；R2/R3仅相差0.44%。

逐Query Mask与目标
- Score/Mask为[N_Q,B]，每Query独立token-budget ranking、skip overflow与稳定Block ID tie-break。
- 前向Hard，反向沿用原implicit budget gate的批量等价导数；cost梯度对全部Query正确累加。
- Attention statistics精确支持[Q,B] Mask，current KV始终全保留。
- update中将所有采样snapshot/step/tile的SSE相加，再除以总dense energy；随后平均模型loss。
- 同时保存每tile ratio均值；评价同时报告Global、tile tails、逐Query tails及胜率，避免混用口径。

Phase0
- 合成干预与数值梯度检查；双GPU各自36个Layer×Head真实BF16检查。
- 不做优化器更新；验证独立Query、预算、Dense/raw gather、正负关系、zero-correction、梯度及H缓存。

Phase1
- 2个Train prompts，generation seed0，3layers×heads[0,5,11]×3histories=54 snapshots。
- 2个Val prompts、相同9pairs、generation seed0、history9360=18 snapshots；无Test tensor访问。
- 共享历史前端全冻结，H只对每snapshot计算一次，复用所有4steps/8tiles；统计缓存在CPU。
- R1/R2/R3同批训练、等权loss；router seeds0/1分别在GPU0/1；各1000更新，warmup50，每100更新Val。
- 每update1snapshot及1tile四steps；无放回snapshot sweeps并轮转8tiles，保证每个tile至少看到两次。
- 稳定/学习门槛：所有记录有限、全部8tiles覆盖、joint Val至少改善1%，各arm最佳共同step Val不恶化超过25%。
- 门槛失败时停止扩展；结果仍用于诊断，不能仅因未过门槛宣称架构无效。

Phase2/3（仅Phase1门槛通过后自动启动）
- 完整16Train/4Val/4Test prompts；generation seeds0/1；全部36pairs、3histories、4steps、8tiles。
- R2/R3共享可训练历史前端，同批等权attention loss，Router seeds0/1/2，训练Keep20%。
- AdamW lr1e-4,wd0.01；4snapshots/update；Warmup500+cosine decay到5%LR；clip1。
- 最少8000、最多20000更新，每1000更新固定36个Val snapshots（每pair1个，覆盖4prompts/2gen-seeds/3histories）。
- best.pt保存两模型Val Global均值的严格最低点，同一时点的partition/H；不分别选择两个不同checkpoint。
- patience的0.2%最小改善幅度使用独立参考值累计，不影响保存更低误差的checkpoint。
- min8000后连续4次Val相对改善不足0.2%停止；达到最大预算只说明预算结束，不证明收敛。
- 每个训练snapshot在min8000时至少9次访问，全部8tiles已覆盖；所选best的曝光应从其sampler另行审计。
- 每个seed完整评估Val/Test/Train全部snapshots与8tiles，在20%主预算及30%/10%跨预算压力测试。
- baselines：同partition逐Query Mass、Spatial+Mass；两者仅离线评估，不作为训练标签。
- 附Hard/Soft、Relevance-only、R3 positive-only与abs冻结干预，报告Mask变化及误差变化。
- 结果按Prompt/Layer/Head/Layer×Head/History/GenerationSeed/Step/Tile切片；3Router seeds报告sample std。

执行与恢复
- reports/plan.json为固定双GPU依赖图；每卡最多1任务，R2/R3共卡。
- python -m streaming_sparse.v05.workflow --prepare 只生成计划。
- python -m streaming_sparse.v05.workflow --execute 执行；非空workflow目录禁止再次启动，避免重复任务。
- 训练可用--resume原run/latest.pt；严格检查source/data/config/GPU，恢复optimizer/sampler/RNG。
- 评估按snapshot原子提交，检查source/checkpoint/data身份后恢复。
- 正式任务源代码指纹固定；代码变动后必须重新跑正确性gate，不能静默混用。
- 全部网络代理变量从任务环境清除；无新下载或Backbone执行。

解释限制
- R0是上一轮历史参考，不能隔离只改变Query粒度的效应；本轮R1同时采用新的Norm/残差/Pre-Q H。
- R1/R2/R3差异与relation冻结干预提供结构诊断；没有额外Norm或去中心重训消融，不能单独证明其因果收益。
- Signed Gram是学习关系，不等同精确Attention Error的真实向量内积。
- 无Oracle/Greedy labels、Mass标签、Diffusion loss、Backbone训练或Selected-set Refinement。
- 不把推理速度、kernel映射或视频质量作为本轮完成条件。
