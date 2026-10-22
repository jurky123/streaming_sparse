Streaming KV Interaction-aware Sparse Attention v0.4
Q Injection Order Ablation & Generalization — 实验准备

1. 当前状态
代码、协议、六组成对配置和双GPU依赖计划已准备。未启动补采、正式训练或Test评估。
原v0.3代码与结果保持冻结。新代码/配置/报告全部位于streaming_sparse/v04/。

2. 已确认的控制变量
用户确认：一套共享可训练的Feature/Clustering/Aggregation，接受A/B attention loss等权平均。
同一个KV snapshot只计算一次F、partition和z，四个diffusion steps和两条分支复用。
A: z -> concat Q Fusion -> 2-layer Interaction -> Linear Score。
B: z -> 同容量2-layer Interaction（不能访问Q）-> 同实现Q Fusion -> Linear Score。
B的query-independent Interaction在snapshot内只计算一次；A按当前query tile计算。
移除独立Pre-Q Encoder；不使用FiLM、Oracle、diffusion loss或额外clustering正则。
Fusion=[z,q,z*q]->Linear384->128+GELU；q为mean-pooling+128->256->128 MLP。
Transformer width128 / heads4 / FFN512 / pre-norm / GELU / dropout0 / 无位置编码。
两条分支全部参数深拷贝同一初始化；其后分别学习。前端是同一个可训练对象。
前端250112参数，每分支746917参数；每个推理模型997029参数，成对训练共1743946参数。
两组容量完全相同；每个Linear/Norm保持原layer×head rank1/affine conditioning。

3. Loss、精度与checkpoint
对每个16-query tile求normalized attention SSE，平均4steps、4snapshots，再平均A/B。
L_pair=0.5*(L_A+L_B)，单个AdamW(lr1e-4,weight_decay0.01)，对整个成对模型clip_norm1。
BF16 autocast用于Q Fusion与Interaction，FP32 master参数；Feature/聚类/聚合、Score Head、
预算和attention statistics/SSE保留FP32，保护分区与ranking精度。A/B精度策略完全相同。
每1000更新存成对checkpoint；主结果固定使用第6000更新，两组必须使用同一前端时点。
不按Test选择模型，也不各自选择不同步数的best而破坏partition identity。

4. 数据与训练矩阵
沿用已固定24 prompts及16/4/4 Prompt split、generation seeds0/1、layers4/14/24、
heads0..11、history4680/9360/14040、steps0..3、8tiles/event、16queries/tile。
目标20736events、5184snapshots、165888 query-tiles（不是165888个单token queries）。
现有4heads可复用：6912events/1728snapshots，真实tensor引用无缺失。
缺heads1/2/4/5/7/8/10/11：13824events/3456snapshots。
缺失数据未补齐前不得把现有4head结果当作本轮正式结果。

训练keep30%、20%分开，各router seeds0/1/2：6个成对训练、12个分支结果。
每个成对任务6000 updates，4snapshots/update，四步用同一tile；评估keep30/20/10%。
sampler在snapshot上无放回循环，对每个snapshot轮转8个tiles，记录访问状态用于精确续训。
完整Train3456snapshots：24000次snapshot/tile访问，相当于最多86.81%的27648个训练snapshot-tiles。
本配置不声称训练遍历了全部8tiles；Train评估单独切片seen/unseen tile，Val/Test仍按unseen prompts评估。

5. 双GPU安排（用户最新要求）
每张卡最多一个进程；一对共享前端+A/B始终同卡，不拆分分支或做DDP。
GPU0: keep30/seed0 -> keep30/seed2 -> keep20/seed1。
GPU1: keep30/seed1 -> keep20/seed0 -> keep20/seed2。
每个任务训练完成后在同一卡顺序评估Train/Val/Test，再开始该卡下一个任务。
两卡补采按generation seed拆分：GPU0采generation seed0，GPU1采generation seed1，
写不同supplemental_generation_seed目录，完成后CPU校验/合并，绝不并发写同一manifest。
两张卡分别运行完整12head正确性检查，其后各自的任务使用匹配硬件/数据/source的gate。
所有任务清空大小写HTTP(S)/ALL_PROXY；资产已在本机，执行计划使用离线模型加载。

6. 指标与证据
主指标Global=sum(SSE)/(sum(dense energy)+1e-8)，不能用每tile相对误差平均替代。
另报tile-error Mean/Median/P90/P99，A/B逐tile配对delta、mean/median improvement、win rate、worst regressions。
按prompt、history、generation seed、layer/head及36组合、step、tile和训练曝光切片。
router seeds0/1/2作为3次重复，以sample std(ddof1)汇总；generation seed是每次训练中的数据维度。
Mass严格使用同一个learned partition；Mass是启发式reference，不是数学最优下界。
每个sample保存相同A/B partition hash、budget/实际retained tokens和真实selected block IDs。
保存partition labels，索引对应不可变raw history的行号；current KV始终完整保留。
Query sensitivity保存同snapshot/step的8tile mask Jaccard与tile误差，不能把mask越不同当作越好。
Interaction诊断包含raw-KV query entropy、history block mass concentration、z余弦redundancy proxy、
score分布及A/B选中token overlap；相关性与分位箱分析只作描述，不作因果归因。
评估以snapshot原子提交，续跑核对checkpoint/source/data身份并避免重复样本。

7. 入口（默认仅查看计划，不运行任务）
streaming_sparse/.venv/bin/python -m streaming_sparse.v04.prepare
streaming_sparse/.venv/bin/python -m streaming_sparse.v04.workflow
实际执行入口为同一个workflow模块的--execute参数；当前没有调用。
workflow首次目录必须是新目录；不会复用旧PID，也不会把已有进程状态当作实验进度。
发生失败后停止启动新任务、保留日志；未提供自动复用旧PID/重启半成品merge的机制。
训练可显式用原run目录与latest.pt续训；评估按已提交snapshot继续。

8. 报告位置
configs/protocol.json：冻结研究协议；configs/keep{20,30}_seed{0,1,2}.json：导出的实际配置。
reports/preparation.json：完整命令、依赖、GPU分配、QKV缺口与磁盘估计。
reports/unit_tests.txt：合成干预/计算/指标/恢复/双进程调度测试（含临时模型微型更新）。
reports/real_partial_cuda_gpu{0,1}.json：现有QKV的BF16零优化器更新检查，full_head_coverage=false。
这些partial checks不能代替补采后每卡36个layer/head组合的正式full12 gate。

9. 解释边界
本轮估计的是共同前端、共同分区上的Q-order效应；共享前端接受两分支共同监督。
本准备阶段没有A优于B的实验结论。最终必须结合unseen prompts、多seed、heads/history/step和tile趋势判断。
训练平均误差、全局能量加权误差、tail与paired win rate应分别报告，不互相替代。
