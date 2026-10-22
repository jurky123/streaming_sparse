# Streaming KV Sparse Routing v0.2 研究报告

报告由已经落盘的结果增量生成，缺失单元格表示尚未完成，不表示零误差。v0.1 原始结果保留，源码定义快照为 `reference_v01.zip`。
本轮主指标先对每个 Prompt 内的各 Layer/Head 分别汇总 SSE / Dense 能量，等权平均 Head 后再跨 Prompt 求均值；同时提供全局 SSE / 能量、Prompt 内能量加权误差与逐 tile Mean/Median/P90/P95。避免高能量 Head 主导主指标，也不能将新的主指标直接与 v0.1 的逐 tile 均值相除。

当前阶段：parallel interrupted；状态：paused。
Supervisor stop marker requested

## P0 诊断

# v0.2 P0 诊断报告

数据沿用既有 4 Prompt × 2 Seed，不重新采集。三个正则权重使用相同初始化、相同采样顺序和优化配置。
所有数值/预算断言在诊断更新上执行；非诊断更新仍检查一热分组、预算、Loss 与梯度有限性。

| λ | 修复比例均值 | 对齐后分区变化 | 分区变化改善误差的更新 | 加权正则/Attention 梯度比中位数 | Val 0.3 | Val 0.2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0133 | 0.0318 | 14/21 | 0 | 0.000505224 | 0.000942394 |
| 0.0001 | 0.0133 | 0.0317 | 16/21 | 0.001005 | 0.000536622 | 0.000938618 |
| 0.001 | 0.0132 | 0.0314 | 11/21 | 0.009635 | 0.00048152 | 0.00087294 |

分区变化使用匈牙利匹配消除任意 Cluster ID 置换。分区改善对照在同一更新后网络参数下，分别用旧/新 Hard Partition 重建 H 和 Mask；实际 token 数同时记录，不能仅根据梯度非零宣称分区改善。
Soft/Hard Mask Gap 固定 Hard Partition；Soft/Hard Clustering Gap 固定同一 Hard Block Mask，只替换统计中的 Assignment，属于独立训练诊断。
Greedy 包含前向贪心及 Mass 初始化的有限轮增删交换；均是启发式，不是严格 Oracle 上界。Recovery 仅在 gap > max(1e-8, 5% E_Mass) 时定义，负值保留。
P0 只有一个初始化 Seed 和有限诊断更新，正则对照用于查错和量级判断，不替代三训练 Seed 的正式消融。


后续正则配置：`{
  "weight": 0.001,
  "source": "P0 existing validation prompts; these prompts belong to later Train split"
}`。

## 数据覆盖

Pilot 与 Full 为嵌套划分：Full Train 0–15 / Val 16–19 / Test 20–23；Pilot Train 0–7 / Val 16–17 / Test 20–21。Seed 0/1 始终留在相同 Prompt split。历史 tensor 存于共享 store，view manifest 不复制 KV。

### pilot

{
  "events": 3456,
  "snapshots": 864,
  "prompts": 12,
  "split_prompts": {
    "train": 8,
    "val": 2,
    "test": 2
  },
  "split_events": {
    "train": 2304,
    "val": 576,
    "test": 576
  },
  "max_backend_error": 3.132060328425723e-06,
  "all_step_positions_consistent": true,
  "complete_cross_product": true
}

### full

{
  "events": 6912,
  "snapshots": 1728,
  "prompts": 24,
  "split_prompts": {
    "train": 16,
    "val": 4,
    "test": 4
  },
  "split_events": {
    "train": 4608,
    "val": 1152,
    "test": 1152
  },
  "max_backend_error": 3.142360583296977e-06,
  "all_step_positions_consistent": true,
  "complete_cross_product": true
}

## 结构与参数控制

{
  "transformer": {
    "encoder_parameters": 1580032,
    "total_parameters": 1913792
  },
  "independent": {
    "encoder_parameters": 1577984,
    "total_parameters": 1911744
  }
}

独立 Block MLP 使用两层 Pre-LN 残差 FFN（256→1536→256），不交换 Block 信息。Frozen 分组复制并冻结完整 KV→group 输入映射，Content 分支仍训练，避免仅冻结 MLP 后位置/ID 映射继续移动分区。Shared A 完全不使用 Layer/Head ID。

## 训练与验证曲线

[full/conditioned/seed0 曲线](runs/full/conditioned/seed0/learning_curves.png)
[full/conditioned/seed1 曲线](runs/full/conditioned/seed1/learning_curves.png)
[full/conditioned/seed2 曲线](runs/full/conditioned/seed2/learning_curves.png)
[full/frozen/seed0 曲线](runs/full/frozen/seed0/learning_curves.png)
[full/frozen/seed1 曲线](runs/full/frozen/seed1/learning_curves.png)
[full/frozen/seed2 曲线](runs/full/frozen/seed2/learning_curves.png)
[full/independent/seed0 曲线](runs/full/independent/seed0/learning_curves.png)
[full/independent/seed1 曲线](runs/full/independent/seed1/learning_curves.png)
[full/independent/seed2 曲线](runs/full/independent/seed2/learning_curves.png)
[full/shared/seed0 曲线](runs/full/shared/seed0/learning_curves.png)
[full/shared/seed1 曲线](runs/full/shared/seed1/learning_curves.png)
[full/shared/seed2 曲线](runs/full/shared/seed2/learning_curves.png)
[full/spatial/seed0 曲线](runs/full/spatial/seed0/learning_curves.png)
[full/spatial/seed1 曲线](runs/full/spatial/seed1/learning_curves.png)
[full/spatial/seed2 曲线](runs/full/spatial/seed2/learning_curves.png)
[pilot/conditioned/seed0 曲线](runs/pilot/conditioned/seed0/learning_curves.png)
[pilot/shared/seed0 曲线](runs/pilot/shared/seed0/learning_curves.png)

## pilot 评估

| Variant | Split | Router Seeds 完成 | Keep | Prompt-macro E | Prompt CI95 | Global E | Tile Mean | Tile P90 | Tile P95 | E / B4 | 实际 Keep |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | train | 1 | 0.3 | 0.0595684 | [0.05449, 0.06711] | 0.0324608 | 0.0614934 | 0.159236 | 0.225043 | 2.451 | 0.29962 |
| shared | train | 1 | 0.2 | 0.0987242 | [0.09146, 0.1078] | 0.0539712 | 0.101728 | 0.264533 | 0.353303 | 2.233 | 0.19966 |
| shared | train | 1 | 0.1 | 0.182405 | [0.1708, 0.1949] | 0.100776 | 0.185826 | 0.480774 | 0.612306 | 1.847 | 0.09968 |
| shared | val | 1 | 0.3 | 0.0611473 | [0.05313, 0.06917] | 0.0296402 | 0.062112 | 0.170684 | 0.236072 | 2.629 | 0.29964 |
| shared | val | 1 | 0.2 | 0.0947846 | [0.0803, 0.1093] | 0.0464322 | 0.0972158 | 0.273253 | 0.365747 | 2.313 | 0.19967 |
| shared | val | 1 | 0.1 | 0.159977 | [0.1353, 0.1846] | 0.0790617 | 0.165661 | 0.469879 | 0.596265 | 1.866 | 0.09970 |
| shared | test | 1 | 0.3 | 0.0744805 | [0.06581, 0.08315] | 0.0408694 | 0.0744336 | 0.19771 | 0.285002 | 2.846 | 0.29966 |
| shared | test | 1 | 0.2 | 0.117386 | [0.1072, 0.1276] | 0.0650618 | 0.116993 | 0.308415 | 0.417993 | 2.486 | 0.19965 |
| shared | test | 1 | 0.1 | 0.201483 | [0.1877, 0.2152] | 0.113368 | 0.200264 | 0.524086 | 0.649244 | 1.938 | 0.09971 |
| conditioned | train | 1 | 0.3 | 0.0375694 | [0.03467, 0.04204] | 0.0188824 | 0.0395447 | 0.106172 | 0.145009 | 1.798 | 0.29963 |
| conditioned | train | 1 | 0.2 | 0.0695891 | [0.06542, 0.07583] | 0.0358923 | 0.0723598 | 0.191624 | 0.255753 | 1.815 | 0.19966 |
| conditioned | train | 1 | 0.1 | 0.144582 | [0.1368, 0.1536] | 0.0775178 | 0.148904 | 0.387891 | 0.501038 | 1.702 | 0.09967 |
| conditioned | val | 1 | 0.3 | 0.0399145 | [0.0355, 0.04433] | 0.0195751 | 0.0411787 | 0.113636 | 0.155494 | 2.026 | 0.29962 |
| conditioned | val | 1 | 0.2 | 0.0687976 | [0.06062, 0.07698] | 0.0334432 | 0.0710226 | 0.195169 | 0.266534 | 1.944 | 0.19966 |
| conditioned | val | 1 | 0.1 | 0.128975 | [0.1119, 0.146] | 0.0632316 | 0.13406 | 0.370895 | 0.487663 | 1.739 | 0.09969 |
| conditioned | test | 1 | 0.3 | 0.0492589 | [0.04471, 0.05381] | 0.0248085 | 0.050499 | 0.131357 | 0.183948 | 2.183 | 0.29965 |
| conditioned | test | 1 | 0.2 | 0.0864613 | [0.07805, 0.09487] | 0.0449608 | 0.0877023 | 0.230987 | 0.304433 | 2.096 | 0.19969 |
| conditioned | test | 1 | 0.1 | 0.169351 | [0.1557, 0.183] | 0.0928342 | 0.172364 | 0.443056 | 0.559443 | 1.867 | 0.09970 |
## full 评估

| Variant | Split | Router Seeds 完成 | Keep | Prompt-macro E | Prompt CI95 | Global E | Tile Mean | Tile P90 | Tile P95 | E / B4 | 实际 Keep |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | train | 0 | — | 未完成 | — | — | — | — | — | — | — |
| shared | val | 3 | 0.3 | 0.0321964 | [0.02972, 0.0368] | 0.0156774 | 0.0342072 | 0.0924601 | 0.133632 | 1.551 | 0.29937 |
| shared | val | 3 | 0.2 | 0.0597866 | [0.05355, 0.06793] | 0.0299479 | 0.0638776 | 0.171492 | 0.238516 | 1.603 | 0.19942 |
| shared | val | 3 | 0.1 | 0.125209 | [0.109, 0.1387] | 0.0647069 | 0.133388 | 0.355321 | 0.467132 | 1.613 | 0.09946 |
| shared | test | 3 | 0.3 | 0.0366331 | [0.03304, 0.04023] | 0.0180998 | 0.0383628 | 0.10397 | 0.140575 | 1.496 | 0.29939 |
| shared | test | 3 | 0.2 | 0.0698601 | [0.06168, 0.07804] | 0.0365652 | 0.0723026 | 0.191488 | 0.248678 | 1.592 | 0.19942 |
| shared | test | 3 | 0.1 | 0.151689 | [0.1325, 0.1708] | 0.0856801 | 0.155346 | 0.396787 | 0.502789 | 1.634 | 0.09946 |
| conditioned | train | 0 | — | 未完成 | — | — | — | — | — | — | — |
| conditioned | val | 3 | 0.3 | 0.0221613 | [0.02034, 0.02541] | 0.0106268 | 0.0238789 | 0.0673115 | 0.0955024 | 1.597 | 0.29930 |
| conditioned | val | 3 | 0.2 | 0.0430432 | [0.03859, 0.04965] | 0.0210887 | 0.0460765 | 0.126596 | 0.178124 | 1.658 | 0.19934 |
| conditioned | val | 3 | 0.1 | 0.0959416 | [0.0838, 0.1099] | 0.0490725 | 0.102125 | 0.276215 | 0.368363 | 1.653 | 0.09937 |
| conditioned | test | 3 | 0.3 | 0.0277935 | [0.02327, 0.03232] | 0.0132516 | 0.0291262 | 0.078348 | 0.107719 | 1.650 | 0.29931 |
| conditioned | test | 3 | 0.2 | 0.0545692 | [0.04575, 0.06339] | 0.028051 | 0.0577869 | 0.148452 | 0.200393 | 1.742 | 0.19934 |
| conditioned | test | 3 | 0.1 | 0.122571 | [0.1034, 0.1418] | 0.0673134 | 0.129003 | 0.325524 | 0.423134 | 1.728 | 0.09937 |
| spatial | train | 0 | — | 未完成 | — | — | — | — | — | — | — |
| spatial | val | 3 | 0.3 | 0.0162641 | [0.01482, 0.01774] | 0.00703442 | 0.0175649 | 0.0498343 | 0.075226 | 1.511 | 0.29885 |
| spatial | val | 3 | 0.2 | 0.0328163 | [0.02907, 0.0358] | 0.0153864 | 0.0350599 | 0.0968226 | 0.14179 | 1.602 | 0.19912 |
| spatial | val | 3 | 0.1 | 0.088598 | [0.07341, 0.1009] | 0.047002 | 0.090971 | 0.247199 | 0.343625 | 1.855 | 0.09856 |
| spatial | test | 3 | 0.3 | 0.0195298 | [0.01726, 0.02077] | 0.00881363 | 0.0214661 | 0.0595684 | 0.0858771 | 1.468 | 0.29886 |
| spatial | test | 3 | 0.2 | 0.0401892 | [0.03644, 0.04338] | 0.0202057 | 0.0432565 | 0.117221 | 0.164646 | 1.612 | 0.19913 |
| spatial | test | 3 | 0.1 | 0.109935 | [0.09787, 0.122] | 0.0656188 | 0.112567 | 0.298854 | 0.409169 | 1.911 | 0.09852 |
| frozen | train | 0 | — | 未完成 | — | — | — | — | — | — | — |
| frozen | val | 3 | 0.3 | 0.0725625 | [0.06293, 0.0822] | 0.0406164 | 0.0737536 | 0.195142 | 0.267677 | 2.723 | 0.29981 |
| frozen | val | 3 | 0.2 | 0.111853 | [0.09586, 0.1257] | 0.0638194 | 0.113847 | 0.298833 | 0.392107 | 2.268 | 0.19986 |
| frozen | val | 3 | 0.1 | 0.191156 | [0.1608, 0.2147] | 0.113518 | 0.195531 | 0.503858 | 0.640693 | 1.763 | 0.09989 |
| frozen | test | 3 | 0.3 | 0.0904968 | [0.07569, 0.1059] | 0.0578931 | 0.0882345 | 0.228113 | 0.315488 | 2.745 | 0.29981 |
| frozen | test | 3 | 0.2 | 0.138099 | [0.1174, 0.1604] | 0.0899071 | 0.136038 | 0.344573 | 0.450385 | 2.264 | 0.19985 |
| frozen | test | 3 | 0.1 | 0.231141 | [0.2026, 0.2642] | 0.154303 | 0.231489 | 0.564239 | 0.698748 | 1.701 | 0.09990 |
| independent | train | 0 | — | 未完成 | — | — | — | — | — | — | — |
| independent | val | 3 | 0.3 | 0.0217951 | [0.01964, 0.02556] | 0.0105092 | 0.0233832 | 0.0658222 | 0.0938366 | 1.593 | 0.29933 |
| independent | val | 3 | 0.2 | 0.0427006 | [0.03854, 0.04895] | 0.0212477 | 0.0456165 | 0.126004 | 0.176441 | 1.655 | 0.19936 |
| independent | val | 3 | 0.1 | 0.0963689 | [0.0844, 0.1104] | 0.0495868 | 0.102383 | 0.278557 | 0.375392 | 1.658 | 0.09937 |
| independent | test | 3 | 0.3 | 0.0268621 | [0.02313, 0.03059] | 0.0131129 | 0.0282469 | 0.0769362 | 0.106504 | 1.611 | 0.29935 |
| independent | test | 3 | 0.2 | 0.0535687 | [0.04552, 0.06162] | 0.0283571 | 0.0557339 | 0.147946 | 0.199713 | 1.720 | 0.19937 |
| independent | test | 3 | 0.1 | 0.121524 | [0.1048, 0.1383] | 0.0690559 | 0.125691 | 0.324914 | 0.422296 | 1.718 | 0.09941 |

## Oracle Headroom

| Stage/Variant/Split | Tiles × Keep | 显著 Headroom 样本 | Mean E_Mass | Mean E_Greedy | Mean Headroom | Recovery 中位数（保留负值） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot/shared/val | 36 | 18 | 0.074629 | 0.0708757 | 0.00375326 | -9.166 |
| pilot/shared/test | 36 | 22 | 0.0607621 | 0.0537696 | 0.00699251 | -32.8 |
| pilot/conditioned/val | 36 | 17 | 0.0712429 | 0.0684334 | 0.00280948 | -2.033 |
| pilot/conditioned/test | 36 | 24 | 0.0573273 | 0.0502178 | 0.00710955 | -19.69 |
| full/conditioned/val | 216 | 139 | 0.0452501 | 0.0389261 | 0.00632396 | -3.812 |
| full/conditioned/test | 216 | 108 | 0.0306138 | 0.0274085 | 0.00320534 | -9.659 |

Greedy 为精确统计下的启发式参考：前向贪心、Mass 初始化的有限轮增/删/交换，与 Mass 本身取最小误差。它不是数学最优上界。只在 G > max(1e-8, 0.05 E_Mass) 时解释 Recovery，负值保留。每条 Oracle 记录保存实际 token 数；未找到改进不能证明不存在更优组合。
B4 使用同一名义 token 预算；B4_cost_capped 使用 Ours 实际 token 数作为预算上限，仍需核对实际利用率。原始逐 tile 文件保存全部实际成本、Hard Mask 与 query 索引。

## 研究决策与当前边界

H1：以 Full 三训练 Seed、独立 Test Prompt 的 Prompt-macro 误差及 B4 比值判断；Pilot 单 Seed 只作稳定性和初步泛化检查。
H2：比较 conditioned / frozen / spatial，并结合 B4 的分区选择能力和 P0 分区单步对照，不能仅用非零梯度作结论。
H3：比较 conditioned Transformer 与参数量相近 independent Block MLP。
H4：优先看 Greedy 绝对改善及预算利用率，再解释 Recovery。
P4 不自动执行。结果接近 Mass 时保持简单 Decoder；主要差距为泛化时研究分布与条件化；训练和验证均差且存在明显 Headroom 时，才逐项考虑 Keep 条件、Query 表示、Query-conditioned Interaction。

完整配置、随机状态与优化器状态保存在各 run；best.pt 按固定分层验证子集选择，final.pt 为最终更新。验证曲线、训练曲线、梯度分解、Assignment 和 Hard Mask 均保存。最终 Val/Test 使用完整 split。
正式主要对照需要 Full 三 Router Seeds 全部完成；缺失结果时不宣称阶段完成或稳定优于 Mass。Test Prompt 仅 4 个，仍属于初步证据。
