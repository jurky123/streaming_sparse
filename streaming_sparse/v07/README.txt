CAVIR v0.7 离线算法实验

原始设计：design/（保留用户 ZIP 内的两份 Markdown、DOCX、README 与输入 SHA256）
执行协议：configs/protocol.json
冻结任务图：reports/plan.json；冻结源码：reports/launched_source/
直观状态：reports/STATUS.txt
命令：streaming_sparse/.venv/bin/python -m streaming_sparse.v07.progress

范围约束（按原设计）：
本轮只验证算法正确性、Attention Output 近似质量、交互建模的可辨识收益及 held-out 泛化。
不要求高效 Sparse Kernel、GPU 延迟、端到端吞吐／加速、完整视频生成质量或 Backbone 共训练。
所有最终误差均对选中原始历史 KV + 全部可见当前段 KV所得到的真实 Hard Mask Attention Output评估。
真实 Dense/Block Statistics只能用于离线监督或上界诊断，不能进入可部署 Router 的评分。

数据与边界：
复用现有原始数据，3456 Train / 864 Val snapshots；36 Layer×Head、两代视频seed、三种history、四步、每步8x16个独立Query。
按 Prompt 切分，因此同一视频的全部step/layer/head/history/query不会跨split。
当前Q/K来自原模型的 QK Norm/RoPE之后；已检查实际 cached-attention调用未传额外scale/bias/causal选项。
当前块内所有被传入的KV对Query可见，不能误施加逐Token三角因果Mask。
显式有效性依据已审计的原始Hook重建；原始Page按历史tensor行顺序64一组，保留末尾不足64的Page。
原始历史/KV不被原型替代；原型只用于决定选哪些原始历史Page。

Test说明：
用户允许在数据可用时复用现有设计。现有原始QKV足以支持本轮Train/Val离线验证。
旧Test已经参与v0.5/v0.6研究设计，不作为v0.7新的独立Test。代码禁止自动读取旧Test。
本队列先完成Train/Val核心验证；独立最终Test需要另外锁定新Prompt后再一次性评估。
因此本队列的完成报告叫Train/Val报告，不冒充完成发表级独立Test。

阶段：
1. Phase0：14项单元检查；实际Backbone checkpoint SHA；真实36pairs、四steps、多预算、非连续Query。
   Dense重建、原始KV gather、删除残差恒等式、当前/历史LogSumExp合并、预算、Mask Forward、梯度全部检查。
   任一失败保存重现数据并停止后续任务。Phase0不做optimizer更新。
2. Phase1：连续Page64固定Partition。T0随机排序装箱、T1真值Mass、T2真值单块误差、T3一次集合重排。
   beta仅在完整Val上从预注册网格选择；给出macro/micro误差、Group bootstrap和交互强度分层。
   T0是均匀随机分数后装箱；变成本时不是所有可行子集上的均匀分布。
   固定36个Val Query额外枚举可行one-swap，仅作诊断，不产生训练标签。
3. Phase2：S1解析均值无可训练参数，作为确定性对照评估一次；S4/S8各3训练seeds。
   真实logZ、weighted value、estimated dense监督；仅Train初始量级用于归一化权重。
   当前模型的Sketch每次更新重新构造，同一snapshot四个steps共享一次构造结果。
   Full Val检查统计充分性；若S4不充分，触发S2/S4-var/S4-K三seed次级训练。
4. Phase3：相同S4初始化分别训练R3（提供当前段信息）和R2（历史only参照，最终真实Attention仍保留当前段）。
   R1为同S4的预测Mass排序；R1不是完全V-free，因为Joint Assignment仍读取V。
   若训练S4-K，其PredictedMass为K-only Assignment/评分诊断，不读取Value做Mass排序。
   冻结同一训练好的R3+Sketch，再独立训练R3-local与R4，小MLP参数量完全相同。
   R4仅在真值one-pass余量和S4充分性都通过时启动；否则继续Local并明确记录R4停止原因。
   R4不是Post-Q Transformer，没有B×B关系矩阵；只计算一次D,z,有限Add/Remove价值，再重新装箱一次。
   R4-shuffle是冻结干预，错配同tile内初始Mask与Query统计量，不作为重新训练正例。
5. Phase4：主训练20%；完整Val评估20/10/30%，10/30是预算外推。
   附固定36-snapshot Train监控集，明确不是全量Train评估。

训练：
Sketch min8000/max12000，基础Scorer min8000/max20000，Correction min8000/max12000。
AdamW lr1e-4，warmup300，cosine到5%，clip1.0；每1000步验证，四次无0.2%改善的平台规则。
每更新4 snapshots、每snapshot四步共用同一tile，按8 tiles轮换；完整覆盖与选中checkpoint覆盖分别记录。
Checkpoint始终保存真实最低Val，平台显著改善阈值只用于停止，不用于丢弃细微最优点。
三seed使用相同数据划分，同seed各模型使用相同采样顺序。
冻结Correction使用独立loss之和与独立梯度裁剪，冻结基础参数SHA在每次验证检查。

数值：
原始QK/原型匹配、评分、真实Mask Output与损失FP32；归一化残差几何、保留/排除项归约及有限变化内部量使用FP64，再导出FP32评分特征；禁用TF32。空原型显式屏蔽，Soft Assignment同时聚合K/V。
真实Hard Output使用logspace重建，无论Logits大小，空历史Mask仍正确恢复当前段输出。
Hard Forward +预算STE；隐式阈值导数包含成本。Mask对输出的解析导数在正常范围精确。
仅对病态极端Lb-logZselected>30做FP32梯度溢出保护；Forward仍精确，保护计数随评估输出。
Staged冻结基础时M0停止梯度；显式joint/scratch模式M0使用相同预算STE，最终Gate也用STE。

指标：
query-micro SSE/energy，(Prompt,Video Seed)group-macro均值、median/P25/P75，query/tile尾部。
Paired CI按video-group固定随机种子bootstrap2000次，不能按单Query当独立样本。
LogZ同时给block-weighted与mass-weighted MAE/RMSE，Mass L1，当前Mass误差、weighted Value、Dense估计误差。
残差方向与低范数计数、singleton log-score相关性及true/predicted singleton mask重合率。
全量Val分层：Prompt/Seed/Layer/Head/Pair/Step/History及History×Step。
保存逐Query原始索引、真实Token保留数、Full评估压缩mask、改Mask后的改善/恶化及Jaccard。

优先级与未启动的次级项：
主队列66节点含条件分支，不代表66项全部一定执行。
page128、joint和scratch有独立显式训练入口，不自动混入frozen_same_base结果。
R0-P0新训练和chunk-local Feature Packing待核心结果明确后实施，本轮启动不声称已完成这些次级项。
旧v0.6 C0只作不同Partition的历史参考，不以其数字直接归因新Router架构收益。

调度：
用户优先v0.7；v0.6派发器已暂停，当前两个任务完成后释放GPU。
调度器检查真实GPU进程，任何卡被现有任务占用就等待，最多每卡一个新实验。
清除http/https/all代理变量，HF离线，全部数据与资产使用本机文件。
任一任务失败后停止派发新任务，不覆盖过去版本的源码或结果。

数值修复记录：Phase0 L4H1曾因FP32相消后除小保留Mass失败，原始尝试与重现样例保存在reports/attempts/initial_phase0_fp32_cancellation。使用D=z*estimatedDense-retainedNumerator及排除自身的前缀/后缀和，保持原始STE代理导数；不放宽5e-5阈值。
