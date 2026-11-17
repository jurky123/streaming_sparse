# 分母/动作评分融合：本轮筛查

当前基线为用户指定旧快速精算＋完整均值/rank8补偿，Risk/摘要/Full_Pset不变；Reader A-only＋split，PAGE32/lazy关闭。输入使用fast_attention_baseline_v1/route_capture/route_entry.pt，12heads×72tiles×16probes×432页。GPU0–7分别运行独占模块实验，最终计时双方交替预热后各50轮。

本轮不推广新候选：完整融合最佳约1.010×，单独预计算第一步约1.002×；没有可稳定兑现的明显收益。默认FUSED_DEN/PRECOMPUTE_DEN均false，旧快速Attention默认保持。

## 理论假设与编译证据修正

原先从表达式估计，每个512动作×16probe重复四次logaddexp；理论上可以把第一步从7077888次减至221184次，少6856704次。但**当前实际Inductor生成代码已经做了这个共享**：第一步融合在221184行的排除候选归约中，随后7077888项kernel只剩3个log1p/exp。因此基线实际已是221184＋3×7077888＝21454848对，不是28311552对。此前将685万次称为待消除重复，是把其他历史缓存版本的生成代码用于当前路径，已纠正。

compiler_evidence/保留生成源码：含221184行归约的一份wrapper明确含一次log1p；动作分母pointwise含3次log1p。生成代码以kernel形状匹配当前执行profile，不以Python表达式推断重复量。

直接融合仍可避免常规动作56.62MB分母写入及再次读取，理论访存下限约0.089ms（1266.4GB/s参考）；但首版把第一步重复带回融合核，寄存器与FP64非线性吞吐抵消收益。增加共享first_log后，最好约1%收益；现有分母＋评分约0.60ms，融合核同量级。小BLOCK/不同warp进一步尝试未出现明显改善，不能把只计访存的理论差距当作可兑现提速。

## 隔离结果

|实验|前/ms|后/ms|加速|最大评分差|动作/no-op|
|---|---:|---:|---:|---:|---|
|kind_a_b128_w4|3.710|4.349|0.8532×|2.78e-17|True/True|
|kind_a_b128_w8|3.704|4.087|0.9064×|1.39e-17|True/True|
|kind_a_b16_w2_shared|3.701|3.674|1.0073×|1.39e-17|True/True|
|kind_a_b16_w4_shared|3.701|3.698|1.0007×|1.39e-17|True/True|
|kind_a_b32_w1|3.715|4.423|0.8400×|2.78e-17|True/True|
|kind_a_b32_w2|3.707|3.944|0.9399×|2.78e-17|True/True|
|kind_a_b32_w2_shared|3.703|3.817|0.9702×|2.78e-17|True/True|
|kind_a_b32_w4|3.714|3.717|0.9993×|1.73e-18|True/True|
|kind_a_b32_w4_shared|3.706|3.669|1.0099×|1.73e-18|True/True|
|kind_a_b32_w8_shared|3.722|3.814|0.9757×|1.39e-17|True/True|
|kind_a_b64_w2|3.703|4.734|0.7822×|2.78e-17|True/True|
|kind_a_b64_w4|3.700|3.932|0.9410×|1.39e-17|True/True|
|kind_a_b64_w8|3.701|3.839|0.9642×|2.78e-17|True/True|
|kind_a_b64_w8_shared|3.704|3.797|0.9753×|2.78e-17|True/True|
|kind_a_b8_w2_shared|3.721|3.759|0.9897×|2.78e-17|True/True|
|kind_a_b8_w4_shared|3.702|3.711|0.9974×|1.39e-17|True/True|
|kind_a_p216_b32_w4_precompute|2.789|2.782|1.0024×|5.55e-17|True/True|
|kind_a_p288_b32_w4_precompute|3.230|3.225|1.0016×|1.39e-17|True/True|
|kind_a_p360_b32_w4_precompute|3.270|3.264|1.0016×|1.04e-17|True/True|
|kind_a_p432_b32_w4_precompute|3.699|3.690|1.0024×|1.73e-18|True/True|

shared表示融合核外先算第一步；precompute表示只手动共享第一步，保留原评分kernel。其余为全部融合。融合只对需要回退的动作写DEN，fallback沿用原正质量重建，不用主导质量直接相减。对实际样本的最高动作/no-op均一致，但评分并非全部逐位一致。

## 数值验收

14份E49回归及主导页移出案例通过（stability/validation.json），覆盖极微小正分；first_den与torch.compile的logaddexp在随机和±inf/±1000差距输入逐位一致。微小完整评分差可能来自编译边界变化；不由端点单项一致推断整条评分逐位一致。

没有做本轮端到端视频：隔离阶段已否决候选，不将微小/噪声级模块收益推进视频验收，也不据此宣称视频相同。新基线的上一轮完整短视频验收仍有效。

## 改动和下一步

scores_den.py为FP64稳定logadd、融合评分与第一步预计算；scores_moments.py和reader_reduce.py仅增加默认关闭的研究开关。check_den_scores.py记录新基线输入上的评分/动作/no-op和profile；check_first_den.py单独验证logadd数学边界。

下一步关注排除候选向量BMM的低吞吐及Reader系数重用；其理论距离必须按实际生成代码和精度核算，避免继续优化编译器已经消除的重复工作。现有完整模块距离表见../fast_attention_baseline_v1/MODULE_DISTANCE.zh.md，本轮没有改变部署性能数据。
