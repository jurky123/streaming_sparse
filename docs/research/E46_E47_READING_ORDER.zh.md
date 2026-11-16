# E46/E47阅读顺序与当前判断

1. [E46报告](evidence/e46_supervision_bridge_v1/REPORT.md)：先看M16/Q64监督对齐，再看冻结小集模型到完整训练/开发事件的桥接，最后比较同预算按层与共享Reader。
2. [E46完整桥接表](evidence/e46_supervision_bridge_v1/bridge_summary.csv)：先筛policy=Set、dimension=overall，再分cohort/layer/head/trajectory；拟合48事件和其余1104训练事件不是独立视频。
3. [E46目标对齐表](evidence/e46_supervision_bridge_v1/small_alignment_summary.csv)：先比较最终2304步，不按开发或中间收益挑选checkpoint。
4. [E47报告](evidence/e47_conditional_value_v1/REPORT.md)：明确真实条件价值分母极小，因此近邻协议主要检验稳定性。
5. [E47条件价值表](evidence/e47_conditional_value_v1/conditional_value.csv)：V_fixed+读取所有状态标签，仅用于离线参照，不是可部署的偏好复用器。
6. 完整矩阵和模型：[E46审阅包](evidence/e46_supervision_bridge_v1/e46_review.tar.gz)、[E47审阅包](evidence/e47_conditional_value_v1/e47_review.tar.gz)。这两份文件已随Git提交，不再依赖仓库未跟踪的artifacts路径。包内包含源码闭包和SHA清单；原始训练数据仍需既有规范缓存。

前置机制：[E43](evidence/e43_fit_v1/REPORT.md) → [E44](evidence/e44_fit_diagnostics_v1/REPORT.md) → [E45](evidence/e45_frozen_replay_v1/REPORT.md)。E45用原E43 P；E47才直接评价E44 Pset，不把两轮拼成未运行的成功链条。

当前结论：目标对齐没有稳定改善小集物理拟合与Q64收益；按层解耦结果不一致，未确认坐标失配是主因；Pset小集高收益不能延伸为全事件可靠性；三个近邻状态所需的真实动作适应空间很小。继续封存Test，不自动增加模型、损失或Query Selector。
