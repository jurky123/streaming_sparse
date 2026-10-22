Streaming KV Sparse Routing v0.6
Relevance 与 Block Interaction 的分离验证

入口：
  streaming_sparse/.venv/bin/python -m streaming_sparse.v06.progress
  直观状态文件：streaming_sparse/v06/reports/STATUS.txt（运行时每15秒刷新）
  总队列：reports/plan.json；任务状态与日志：reports/workflow/
  预注册方案：configs/protocol.json；固定源码：reports/launched_source/

数据与限制：
  复用 v0.4/v0.5 已捕获的全部逐 Query QKV，不重新 capture。
  3456 Train / 864 Val / 864 旧开发 Test snapshots，共5184。
  24 prompts，generation seeds0/1，layers4/14/24，heads0..11，三种 history，四步，每步8x16 queries。
  Router seeds0/1/2。训练 Keep20%；评估30/20/10%，30/10为跨预算测试。
  Backbone frozen。无 Mass/Oracle 标签，无 diffusion loss，无 selected-set refinement。
  旧 Test 参与过研究设计，只是开发测试；发表级结论仍需新的独立 Test prompts。

训练顺序：
  Stage I：从新初始化独立训练 C0，模型不包含 Correction 参数。
    AdamW lr1e-4，4 snapshots/update，500 warmup，cosine decay，clip1.0。
    最少8000、最多20000 updates；每1000步检查固定36-snapshot Val 全8tiles/4steps。
    连续4次无>=0.2%显著改善后按平台规则停止；真实最低 Val 独立保存，不受显著阈值影响。
  Stage II：同一个 seed 的 C1/C2/C3 严格加载同一个 C0 best。
    冻结 F/Partition/Aggregator/Pre-Q H/Q Fusion/Relevance，包括所有 PairNorm 与残差。
    每次验证检查基础参数完整 digest；所有分支读取相同 fusion 与 relevance 张量。
    C1 是严格逐 Block 的 gated MLP，不能读取其他 Block；参数量与 C2 完全相同。
    C2 普通2L Attention；C3 原始2L centered-cosine signed interaction。
    每支 Correction 初始化alpha=0.1，独立梯度裁剪与 Adam moments。
    合并执行采用独立 loss 之和，冻结基线不接收梯度；已验证与单独训练的更新逐位相同。
    最少8000、最多12000 updates，warmup300；每支独立选最佳 checkpoint、独立平台停止。
    有界CPU LRU仅保留8个历史 snapshot，避免把全数据的 QKV 和统计量无界留在内存。
  Stage III：只按完整 Val 决策，规则提前写入 protocol。
    至少2/3 seeds 的完整 Val 优于 C0，平均相对改善>=1%，且至少2/3 seeds 的末3次 monitor 中至少2次改善。
    对满足条件的 C1/C2/C3 进行低学习率联合微调：lr1e-5，min4000/max8000。
    未满足条件的 Stage III/Scratch 在队列中明确标为 skipped，不偷偷扩展。
    Interaction 独立价值还必须优于 C1；通过 C0 门槛本身不等于证明跨 Block 交互有效。
  Scratch：仅针对进入 Stage III 的结构，从保存的新初始化 C0 与相同初始 Correction 开始。
    总更新数匹配该 seed 实际 StageI+StageII+StageIII 的总计算预算；不提前停止。
    分段 LR、sampler reset、optimizer reset 匹配三个阶段边界，整个过程一直联合训练。
    同时报告实际训练预算和选中 checkpoint 曝光量；最佳步不等于实际计算成本。

Signed 诊断：
  若 C3 不满足上述稳定条件，对三个 seeds 实际重训练 no_center / amplitude / positive_only。
  原始 centered cosine 已在核心 C3 中训练。其他变量相同，保持原 clustering 和 repair。
  amplitude 使用 centered raw dot/sqrt64，保留向量相对模长，仍用abs-row-sum规范化。
  不以冻结负关系截断代替重训练，不据旧 Test 选择消融。

离线诊断：
  使用预先按 metadata 固定的36个 Val Query，覆盖全部 layer/head pair，保持各自 learned partition/budget。
  比较 C0/C1/C2/C3、Mass、ApproxGreedy。
  ApproxGreedy 仅逐次加入可负担、使真实 SSE 最低且确有改善的 Block；允许不足预算。
  它不是全局最优 Oracle，也不是 learned router 的 selected-set refinement，不给训练提供标签。
  真实交互项 a_b = u_b - z_b*o_dense 的 off-diagonal Gram。
  该 Gram 对应 omitted-set 误差分子；完整误差仍取决于 retained partition function 的平方。
  对每个 Query 数值验证完整公式；记录 Pearson/Spearman/sign agreement 和原始矩阵 NPZ。

结果与审计：
  每个 snapshot 只计算一次 H，四个 diffusion steps 及所有tiles复用。
  以 pooled SSE / pooled dense energy 为主指标，同时记录 Tile mean 和 Query/tile tails。
  Full Val/开发 Test：三预算、全量 snapshot，Paired wins/ties、分层误差、hard/soft、Mass、SpatialMass。
  保存逐 Query 原始 indices、压缩 block masks、block sizes、token-to-block labels、partition hash。
  Correction 改 Mask 后的改善/恶化条件比例单独报告。
  Stage III/Scratch 的 C0 字段指其当前 relevance 分支；独立 C0 正式基线在 frozen_core 结果中。
  不把联合微调后的不同 partition 误报为 StageII 的相同 partition。

运行策略：
  两张 A100，每卡最多一个实验进程；前置18项单元/回归测试及两卡各36 pair真实QKV检查。
  所有网络代理变量在启动器和每个 worker 中清除；使用本地数据与 assets，HF离线模式。
  任一任务失败即停止发出新任务；不覆盖以前版本或重用旧PID。
  83个预注册任务节点包括条件节点；实际执行数由 Val 门槛决定，skipped不是未完成错误。
