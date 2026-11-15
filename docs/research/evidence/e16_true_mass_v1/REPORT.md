# E16：真实 Mass 与摘要 Mass、Risk、集合搜索对照

**完成144个冻结 head–tile，其中 horse_field 两seed的72例作为原评价集；没有新视频、训练或全局求解。** Q64/Page64、R4/rank8、固定10%精确预算、own保护与同状态原始输入全部不变。本轮回到原始基线状态，不使用 E13/E14 的共同交换面板或994个反事实集合。

## 核心结果

评价集在相同16个 probes 下，真实 Mass 相比摘要 Mass 平均逐例 NMSE 改善 **23.89%**，相比当前 Risk 改善 **12.90%**。但从真实 Mass 继续用真实混合输出误差搜索集合，还能改善 **32.53%**；即使用全部64 Query的真实 Mass，仍有 **26.61%** 的交换收益。

这说明：**更准确的质量信息确实有价值，但真实 attention mass 本身仍不是输出近似误差的充分评分。** 也不能把剩余差距全部归因为显式集合条件信息；集合搜索还读取完整有符号残差、V信息和变化分母，本对照没有隔离这些因素。

## 公平定义

真实质量为 m_qp=Z_qp/Σ_p Z_qp，然后在固定 probes 上平均。按平均质量选全候选页top-K，并强制 own。不是直接平均未归一化 Z，不使用摘要分母归一化真实 mass。

- summary_mass16：保存的真实在线摘要 Mass mask。
- risk16：保存的当前在线四项 Risk mask。
- true_mass16：真实 KV 质量，使用与在线方法完全相同的16个 probes。
- true_mass64：真实质量使用全部64 Query；单列信息上限对照，不与16 probes混称同信息量。
- summary_mass16/64_fp64：相同 FP64 页贡献模型下的摘要 Mass 对照。所有144例中，16版mask与保存的部署mask完全相同。
- greedy、risk/mass/greedy_exchange：复用前轮集合搜索mask，重新计算真实目标。
- true_mass16/64_exchange：本轮新跑，从真实 Mass 初始化，枚举全部合法一进一出交换直到局部收敛；不缩小候选池。分别为P216/K21或P410/K41。

所有最终指标均为全部64 Query、固定摘要、真实变化分母下的 pre-W_O 混合输出 NMSE。真实 Mass 需要读取原始 KV，是离线 Oracle 信息，不能宣称可直接在线获得。

## 评价集主表

| 方法 | 平均 NMSE | 相对 Risk 的平均逐例改善 |
|---|---:|---:|
| summary_mass16 | 0.01193250 | -24.78% |
| risk16 | 0.01027546 | +0.00% |
| true_mass16 | 0.00903227 | +12.90% |
| true_mass64 | 0.00847474 | +17.14% |
| greedy | 0.00550884 | +40.18% |
| risk_exchange | 0.00545985 | +40.48% |
| true_mass16_exchange | 0.00546124 | +40.48% |
| true_mass64_exchange | 0.00546107 | +40.49% |

表中最后一列是 mean(1−J_method/J_risk)，不是1−mean(J_method)/mean(J_risk)。两种统计都保留在CSV，不能混用。

### 配对比较

| 基线 → 对照 | 平均逐例改善 | 中位数改善 | 改善／退化／平局 |
|---|---:|---:|---:|
| 摘要Mass16 → 真实Mass16 | 23.89% | 15.54% | 61／3／8 |
| Risk16 → 真实Mass16 | 12.90% | 8.74% | 49／22／1 |
| 真实Mass16 → 真实Mass64 | 7.01% | 1.76% | 41／14／17 |
| 真实Mass16 → 从它出发Exchange | 32.53% | 31.42% | 71／0／1 |
| 真实Mass64 → 从它出发Exchange | 26.61% | 26.46% | 69／0／3 |

真实Mass并不是每例都优于Risk；它在22例上更差。增加Query覆盖也不保证混合输出NMSE单调改善，因为优化mass总量和优化输出误差是不同目标。

相同64 probes下，真实Mass64相对摘要Mass64的平均逐例改善为26.23%，因此主要结果不只是Query覆盖差异。原Risk+Exchange相对真实Mass16/64仍分别改善32.53%/26.61%，与从真实Mass重新搜索的结果接近。

全部144例中，真实Mass16与Risk出发的Exchange最终mask有123例完全相同；其余不得称同一最优解。搜索终点只取得 **一次交换局部容差最优**，本轮没有重新认证全局OPT。

## 机制解释与限制

真实Mass准确描述某页在Dense中的注意力质量，但不直接描述它的摘要误差：高mass页若摘要准确，精算收益可能小；低mass页若Value误差或残差累积较大，仍可能值得精算。上面是与目标结构一致的解释，本轮对照本身没有分别量化这些因果成分。

结果支持把“质量估计准确度”和“误差目标匹配度”分别研究。真实Mass缩小了当前独立评分与集合搜索之间的差距，但未消除它。不能从本结果直接推导必须训练集合网络，也不能声称Mass可恢复某个全局OPT比例。

数据来自此前使用过的city_walk/horse_field，各自分列；主评价72例是原分割，不构成新场景泛化证据。没有闭环视频、部署延迟或训练结果。原数据、摘要、骨干均冻结。

## 验证与交付

- 原数据734文件哈希保持；源搜索文件哈希记录在每例result.json。
- 真实质量与独立 exact_page_logz softmax 重建最大差 1.221e-15。
- 每个mask核查精确页数与own；真实Mass达到相应16/64质量目标的约束top-K，不冒称达到NMSE最优。
- 新搜索288次，最终所有合法邻居用直接FP64残差独立验证；Gram与直接目标最大差 3.331e-16；接受轨迹严格下降。
- 旧方法目标独立重算与前轮结果一致。

文件：per_instance.csv、summary.csv、paired_comparisons.csv、stratified.csv、exchange_trajectories.csv、protocol.json、audit.json。events/*保存全部mask、mass分数、轨迹与局部证书；精简包包含这些小文件，不重复打包原始贡献。

在具备原冻结数据的仓库根目录运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m local_experiments.e16_true_mass.run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m local_experiments.e16_true_mass.report
```

默认16个并行worker。代码依赖仓库原set_oracle_v1和E12目标函数；不修改输入数据。改变实现时须使用新的输出版本目录，避免复用旧缓存。
