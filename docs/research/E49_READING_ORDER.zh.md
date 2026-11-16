# E49阅读顺序

1. [报告](evidence/e49_numeric_v1/REPORT.md)：先看“所有原提议均区间正号、实际损害不变”，再看高精度案例。
2. [汇总表](evidence/e49_numeric_v1/summary.csv)：Set/Independent、历史/统一/增量/区间协议均同范围比较，先筛overall，再按layer/head/trajectory。
3. [尾部高精度](evidence/e49_numeric_v1/reference_17_tails.csv)、[seed29](evidence/e49_numeric_v1/reference_29_tails.csv)：raw单位不使用Dense阈值，不能把极小值直接称ULP噪声。
4. [同mask与输出变化](evidence/e49_numeric_v1/output_change_reference.csv)：旧两种归约确有误差，但统一归约未改变本次所有最高动作。
5. [审计](evidence/e49_numeric_v1/audit.json)、[平台](evidence/e49_numeric_v1/platform.json)、[协议](E49_NUMERIC_SEMANTICS_PLAN.zh.md)。区间保证只覆盖固定输入与所列平台/算术假设，不是模型安全保证。
6. [高精度输入与源码包](evidence/e49_numeric_v1/reference_review.tar.gz)，完整评分分8包，见[包清单](evidence/e49_numeric_v1/packages.json)。这些包已放仓库证据目录，包含SHA；原始训练缓存仍另需。

本轮结论：两个原E29灾难正分在高精度下仍为正；剩余高损害应首先归模型判断，不以更宽的“数值epsilon”消除。Test继续封存，不训练新摘要或Reader。
