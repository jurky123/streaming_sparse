# E47：Pset冻结E45回放与真实条件价值

沿用E45全部48事件、3个锁定状态、原候选池及no-op规则；仅替换模型为E44最终Pset。无训练或新状态。原E43 P作冻结参照，Pset原96状态矩阵需匹配E44。

每事件状态等权：V_adaptive=sum_s w_s max(0,max_j g_sj)，V_fixedplus=max_j sum_s w_s max(0,g_sj)。两者差为排除接受判断后改变动作的真实额外空间。按all3、trained2、risk_holdout分列；只作为离线解释分母，不混作模型收益可加分解。

保存144状态动作/端点、完整矩阵、真值、mask、Pset移植一致性，结果与原E45按同口径比较。Test封存；不根据结果增选集合或Query。
