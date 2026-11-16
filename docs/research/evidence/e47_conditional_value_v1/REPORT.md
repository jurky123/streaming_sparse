# E47：P_set冻结回放与真实条件价值分母

本轮只使用E44最终P_set，沿用E45锁定的48事件、96训练状态及48邻近留出状态；原E43 P直接重放作参照。无训练、新状态、新Query或Test评价。候选与真实收益矩阵逐位匹配E45；P_set在原96状态的评分逐位匹配E44最终矩阵。

V_adaptive对每个事件的状态等权，逐状态选择真实最佳动作/no-op；V_fixed+选一个固定动作，但每个状态允许真实门控拒绝。差值仅度量排除接受判断后的“改变动作”价值，是离线教师参照，不能部署。不把它当模型收益的可加分解，也不拿不同状态的恢复率直接平均。

|seed|状态范围|V_adaptive|V_fixed+|V_conditional|占真实自适应收益|有条件价值事件|
|---|---|---:|---:|---:|---:|---:|
|17|all3|0.2487|0.246151|0.00254975|1.025%|11/48|
|17|trained2|0.243868|0.243671|0.000197228|0.081%|8/48|
|17|risk_holdout|0.250624|0.248619|0.00200506|0.800%|6/48|
|29|all3|0.286579|0.285849|0.000730354|0.255%|14/48|
|29|trained2|0.284082|0.28405|3.1394e-05|0.011%|10/48|
|29|risk_holdout|0.287519|0.286531|0.000988189|0.344%|9/48|

|seed|模型|状态|策略|恢复率|H_loss|退化|P10|
|---|---|---|---|---:|---:|---:|---:|
|17|E43_P|Risk|Set|38.04%|0.000500|13/48|-2.69%|
|17|E43_P|Risk|Independent|31.35%|0.001164|14/48|-1.44%|
|17|E43_P|Risk|Reuse_other|37.68%|0.000783|14/48|-3.90%|
|17|E43_P|BgSwap_train|Set|37.69%|0.000925|13/48|-4.03%|
|17|E43_P|BgSwap_train|Independent|31.01%|0.001143|16/48|-1.52%|
|17|E43_P|BgSwap_train|Reuse_other|37.85%|0.000629|13/48|-3.26%|
|17|E43_P|BgSwap_holdout|Set|37.06%|0.000589|11/48|-3.93%|
|17|E43_P|BgSwap_holdout|Independent|29.06%|0.001157|14/48|-1.83%|
|17|E43_P|BgSwap_holdout|Reuse_other|36.54%|0.000550|13/48|-3.01%|
|17|E43_P|trained_combined|Set|37.86%|0.001425|26/96|-3.77%|
|17|E43_P|trained_combined|Independent|31.18%|0.002307|30/96|-1.48%|
|17|E43_P|trained_combined|Reuse_other|37.77%|0.001412|27/96|-3.77%|
|17|E44_Pset|Risk|Set|61.44%|0.000416|11/48|-3.77%|
|17|E44_Pset|Risk|Independent|54.90%|0.000617|11/48|-4.69%|
|17|E44_Pset|Risk|Reuse_other|61.10%|0.000712|12/48|-4.53%|
|17|E44_Pset|BgSwap_train|Set|60.98%|0.001122|11/48|-3.92%|
|17|E44_Pset|BgSwap_train|Independent|54.39%|0.000882|12/48|-4.50%|
|17|E44_Pset|BgSwap_train|Reuse_other|61.25%|0.000790|11/48|-3.60%|
|17|E44_Pset|BgSwap_holdout|Set|64.33%|0.000556|10/48|-3.17%|
|17|E44_Pset|BgSwap_holdout|Independent|51.69%|0.000614|11/48|-4.32%|
|17|E44_Pset|BgSwap_holdout|Reuse_other|64.34%|0.000263|10/48|-3.58%|
|17|E44_Pset|trained_combined|Set|61.21%|0.001538|22/96|-3.94%|
|17|E44_Pset|trained_combined|Independent|54.65%|0.001499|23/96|-4.57%|
|17|E44_Pset|trained_combined|Reuse_other|61.18%|0.001502|23/96|-4.58%|
|29|E43_P|Risk|Set|71.44%|0.000433|9/48|-3.73%|
|29|E43_P|Risk|Independent|65.01%|0.000929|10/48|-2.94%|
|29|E43_P|Risk|Reuse_other|57.38%|0.000431|9/48|-1.84%|
|29|E43_P|BgSwap_train|Set|57.45%|0.000470|8/48|-2.00%|
|29|E43_P|BgSwap_train|Independent|64.12%|0.001259|13/48|-3.15%|
|29|E43_P|BgSwap_train|Reuse_other|69.81%|0.000472|9/48|-4.00%|
|29|E43_P|BgSwap_holdout|Set|72.09%|0.000575|9/48|-4.43%|
|29|E43_P|BgSwap_holdout|Independent|66.20%|0.001043|11/48|-3.59%|
|29|E43_P|BgSwap_holdout|Reuse_other|71.77%|0.000428|9/48|-4.15%|
|29|E43_P|trained_combined|Set|64.43%|0.000903|17/96|-3.34%|
|29|E43_P|trained_combined|Independent|64.56%|0.002187|23/96|-3.03%|
|29|E43_P|trained_combined|Reuse_other|63.61%|0.000903|18/96|-3.34%|
|29|E44_Pset|Risk|Set|75.28%|0.000171|9/48|-3.06%|
|29|E44_Pset|Risk|Independent|69.75%|0.000103|8/48|-3.23%|
|29|E44_Pset|Risk|Reuse_other|60.68%|0.000171|9/48|-3.06%|
|29|E44_Pset|BgSwap_train|Set|60.73%|0.000180|10/48|-3.04%|
|29|E44_Pset|BgSwap_train|Independent|69.90%|0.000106|9/48|-1.96%|
|29|E44_Pset|BgSwap_train|Reuse_other|75.02%|0.000180|11/48|-3.04%|
|29|E44_Pset|BgSwap_holdout|Set|75.41%|0.000234|10/48|-5.83%|
|29|E44_Pset|BgSwap_holdout|Independent|70.34%|0.000104|9/48|-2.19%|
|29|E44_Pset|BgSwap_holdout|Reuse_other|75.70%|0.000173|10/48|-2.87%|
|29|E44_Pset|trained_combined|Set|67.99%|0.000351|19/96|-3.25%|
|29|E44_Pset|trained_combined|Independent|69.83%|0.000209|17/96|-3.09%|
|29|E44_Pset|trained_combined|Reuse_other|67.87%|0.000351|20/96|-3.25%|

完整动作索引/端点、评分、true gains及mask均保存。Reuse_other接受判断也来自偏好源状态，不能用目标真值挑动作。Independent与Set使用相同模型信息。V_conditional很小时，本协议主要是邻近稳定性检查；不能据小增量否定其他大范围集合条件价值。V_conditional明显而模型收益差时，也须区分页面预测、动作提议与门控，不直接新增模块。

P_set输出是结构化决策代理，未默认其响应和β物理校准。所有结果限于已见Query/页面和同候选池，不称跨Query或跨轨迹泛化。Test封存。

## 最终判断

- Pset在E45邻近留出Set恢复64.33%/75.41%；沿用Risk偏好为64.34%/75.70%。提高的小集决策能力能够延续到这个邻近集合，但重评分没有额外净收益，且损害略高。
- trained2条件价值只占真实自适应收益0.081%/0.011%；all3为1.025%/0.255%。这说明协议本来就几乎不需要改变动作偏好，不能用它强力验证或否定复杂集合条件策略。
- 原Pset在两个训练状态上Set相对Independent仍是一正一负；本轮没有把小集信息改善自动归为集合交互创新。
- 不增加状态、Query或训练，不根据结果改接受规则。后续若需要检验更强条件学习，应另行预登记有代表性的状态范围，不能在本轮标签上追选“强交互”事件包装为泛化。

审阅包e47_review.tar.gz已实际纳入仓库证据目录，含完整scores.npz和对应冻结模型，避免再次提供仅存在于本地artifacts的链接。
