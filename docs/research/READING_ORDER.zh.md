> **2026-11-18 当前入口：** [仓库现状与工作交接](REPOSITORY_HANDOVER_20261118.zh.md)。包含最新三档基线、已完成质量验证、两轮优化验收、代码地图、复现和暂停点。下文保留各阶段历史记录。

# 理解当前研究路线的文件阅读顺序

不必从E11逐个读到E38。建议先掌握当前结论，再理解误差结构和在线方法，最后回看失败分支。**已完成实验截至E42：固定局部监督覆盖对照。没有新增训练或在线模块，Test封存。**

## 1 最短阅读路线

按下面九步即可理解“在做什么、为什么困难、下一步验证什么”。

1. [截至E37的结论和问题分析](E11_E37_FINDINGS_AND_NEXT_STEPS.zh.md)：先读第1、2、4、5节。区分优化空间、教师表达能力、在线可预测性，明确目前没有通过可靠性门槛的在线修正器。
2. [E18残差来源](evidence/e18_residual_sources_v1/REPORT.md)：理解Value响应与质量误差的强抵消，为什么方向相关和单项回归精度不够。
3. [E29局部响应Reader](evidence/e29_local_reader_v1/REPORT.md)：理解当前实际模型、解析全局中心、Set/Independent/no-op，以及真实β替换的意义。
4. [E37共享β校准](evidence/e37_shared_beta_v1/REPORT.md)：理解为什么解释大部分加权β误差仍没有恢复大部分决策收益。
5. [E38执行报告](evidence/e38_action_attribution_v1/REPORT.md)：先读端点修复与重新决策的区别，再读四角色校准与C/D停止条件；需要核对设计时再读[实验方案](E38_ACTION_ATTRIBUTION_PLAN.zh.md)。

6. [E39执行报告](evidence/e39_action_margin_v1/REPORT.md)：先看A的三项闭合与偏好正确计数，再比较最终T1/T0的实际收益、H_loss和训练监测。不要按epoch0挑选模型。

7. [E40尺度与可辨识性报告](evidence/e40_reader_scale_v1/REPORT.md)：区分理论有界、实际饱和碰撞、隐式尺度数值条件和描述性关联，避免把相关性当成修改Reader的收益证明。

8. [E41训练信号审计报告](evidence/e41_training_signal_v1/REPORT.md)：先看完整训练集真实收益，再区分局部响应覆盖缺口、β梯度覆盖和共享参数冲突；注意参数子样本只覆盖head6。

9. [E42固定权重对照报告](evidence/e42_local_coverage_v1/REPORT.md)：检查“监督份额提高→响应误差小幅改善→实际收益不一致”的证据链，区分复用T0的续训效果和唯一权重修改的增量。

读完后应能回答：**我们不是在找更大的搜索器，而是在判断什么可在线获取的信息能可靠区分有利与有害交换。**

## 2 完整机制阅读路线

| 顺序 | 文件 | 阅读重点 |
|---|---|---|
| 1 | [E11至E29路线总表](E11_E29_EXPLORATION_ROADMAP.zh.md) | 快速浏览逐轮动机，先不展开所有数据表 |
| 2 | [E14集合价值与可观测性](evidence/e14_value_observability_v1/REPORT.md) | 集合条件增量与真实信息优势不能混为一谈 |
| 3 | [E16真实Mass对照](evidence/e16_true_mass_v1/REPORT.md) | 质量更准确仍不等于输出误差最小 |
| 4 | [E17交换收益分解](evidence/e17_swap_decomposition_v1/REPORT.md) → [E18残差来源](evidence/e18_residual_sources_v1/REPORT.md) | I−B+D是收益结构，V/Z是残差来源；两种分解不同 |
| 5 | [E24联合代表](evidence/e24_joint_representatives_v1/REPORT.md) → [E25能力阶梯](evidence/e25_decision_diagnostics_v1/REPORT.md) | 初选提升与集合收益分开；低维教师有效不代表能预测 |
| 6 | [E26系数Oracle](evidence/e26_oracle_coefficients_v1/REPORT.md) → [E27自适应Query](evidence/e27_adaptive_query_v1/REPORT.md) | 区分Query函数表达、页面系数预测、模型实际执行 |
| 7 | [E28局部目标](evidence/e28_local_response_v1/REPORT.md) → [E29在线Reader](evidence/e29_local_reader_v1/REPORT.md) | 为何预测局部响应、解析组合全局中心；教师复用仍非在线 |
| 8 | [E30 β诊断](evidence/e30_beta_diagnostics_v1/REPORT.md) → [E31跨Query复用](evidence/e31_beta_reuse_v1/REPORT.md) | 事件内拟合、Shared、LOTO与严格过去chunk之间的差别 |
| 9 | [E35候选召回](evidence/e35_candidate_recall_v1/REPORT.md) → [E36 Query覆盖](evidence/e36_query_coverage_v1/REPORT.md) | 提议、执行、Top-L和Query覆盖分别有什么限制 |
| 10 | [E37共享校准](evidence/e37_shared_beta_v1/REPORT.md) → [E38执行报告](evidence/e38_action_attribution_v1/REPORT.md) | 从回归指标转向动作间隔错误，按B→A→条件C/D推进 |

## 3 按问题查阅已停止的分支

| 想理解的问题 | 阅读文件 | 应带走的结论 |
|---|---|---|
| 为什么不直接用摘要Value方向评分 | [E15](evidence/e15_summary_information_v1/REPORT.md)、E17、E18 | 方向代理不等于净误差，幅度和抵消也重要 |
| 为什么不继续二阶响应sketch | [E19](evidence/e19_response_moments_v1/REPORT.md)、[E20](evidence/e20_compressed_response_v1/REPORT.md)、[E21](evidence/e21_key_diagnostic_v1/REPORT.md) | 完整统计的两例正信号没有稳定压缩为经济表示 |
| 为什么不直接增加真实KV探测 | [E22](evidence/e22_active_probing_v1/REPORT.md)、[E32](evidence/e32_candidate_key_probe_v1/REPORT.md) | 新信息有价值，但读取开销和错误执行仍高 |
| 为什么不直接拟合更准确的logZ | [E33-A](evidence/e33_mass_representatives_v1/REPORT.md) | 摘要质量、Value混合和Risk路由强耦合 |
| 为什么完整Key均值方差也不够 | [E34](evidence/e34_key_cumulants_v1/REPORT.md) | 二阶累积量大logit失真，不只是低秩投影问题 |
| 为什么双视图不能直接当误差 | [E23](evidence/e23_dual_representatives_v1/REPORT.md) | R4−R2不保证反映Raw64−R4 |

这些结果限定于所测配置，不应被推广为所有低秩、所有抽样或所有R4构建方法都不可能成功。

## 4 需要看实现时的代码顺序

先读报告，再按以下顺序定位代码；不要一开始从捕获视频和全部训练脚本读起。

1. [E24 core.py](../../local_experiments/e24_joint_representatives/core.py)：摘要贡献、真实/预测集合目标及基础数据结构。
2. [E29 core.py](../../local_experiments/e29_local_reader/core.py)：`LocalReader`与`prepare_local`，局部响应目标、摘要中心和训练/评价入口。
3. [E30 run.py](../../local_experiments/e30_beta_diagnostics/run.py)：`local_scores`最关键，稳定正混合的Set评分和同信息Independent；再看教师β替换及质量指标。
4. [E37 run.py](../../local_experiments/e37_shared_beta/run.py)：`fit`的mass²偏移、候选分组、冻结状态与真实收益对照。
5. [E37 report.py](../../local_experiments/e37_shared_beta/report.py)：汇总恢复、逐例指标、误差分解、条件相关和门槛；确认“成功”的具体定义。
6. [E37 audit.py](../../local_experiments/e37_shared_beta/audit.py)：独立长精度重算、原动作重放、源哈希及count/own检查。
7. 若要回到Oracle：读[集合问题定义](../../local_experiments/set_oracle_v1/problem.py)与[集合搜索](../../local_experiments/set_oracle_v1/search.py)，再查早期认证，不必先阅读求解器细节。

E38—E41均已有运行代码和证据。E41实现见[审计入口](../../local_experiments/e41_training_signal/run.py)与[独立核验](../../local_experiments/e41_training_signal/audit.py)，读取冻结缓存，无优化器更新。

## 5 阅读结果表时的核对顺序

1. 先确认数据范围、摘要版本、Risk集合、候选池与评价Query是否相同。
2. 找实际Set、Independent和no-op，再看教师；先看执行，再看事后真门控和Top-L。
3. 同时看汇总收益、逐例平均、退化比例、P10、最差绝对NMSE，不能只挑一个总分。
4. 检查按轨迹和layer的结果，尤其不要让layer29的加权平方误差代表全部事件。
5. 最后看模型成本和门槛；“教师通过”不等于“可部署成功”。

[证据索引](evidence/README.md)收录报告和小型结果表，[快照清单](evidence/manifest.json)提供原artifact路径与哈希。原始QKV、模型检查点和大数组不随这份文档提交，仅凭克隆仓库不能完整重现训练。

## 6 E38内部建议阅读顺序

B1统一动作身份与单位 → B2/B3分支和页面归因 → B4重新决策 → B5响应/覆盖/投影残差 → A归一化敏感度和六配置 → C启动条件与有限支持界 → D训练集校准与提议上限。

[旧对比β方案](E38_CONTRAST_BETA_PLAN.zh.md)仅供追溯设计变化，不属于当前必跑清单。E38主方案里的数值边界和待定校准细则要在执行前锁定，不能根据开发结果补选。

E43：[小型可拟合性对照](evidence/e43_fit_v1/REPORT.md)。先看P/L最终收益及信息误差是否同步，再读身份与条件评分审计限制。

E44：[E43补全与损失诊断](evidence/e44_fit_diagnostics_v1/REPORT.md)。先看六分支实际收益及物理误差，再看Layer29四格教师和条件变化，最后核验身份/Adam时钟。
