# 当前组合的 GPU kernel 成本分解

通过 CPU sparse 标记范围内的 CUDA runtime 和 driver correlation 关联 GPU kernel。仅计算 kernel 持续时间；不把 profiler 下 CUDA event 包含的排队间隙视为算子计算。

| 模块 | 平均 kernel 总时长/ms | 全视频次数 | 外推/s |
|---|---:|---:|---:|
| sparse/current_summary | 0.1133 | 4524 | 0.5126 |
| sparse/route | 0.2835 | 4524 | 1.2824 |
| sparse/exact | 0.2421 | 4524 | 1.0954 |
| sparse/compensation | 0.1945 | 4524 | 0.8798 |
| sparse/history_commit | 0.9932 | 1189 | 1.1809 |

合计外推 4.9511 秒。来自单个稳态 chunk6；早期形状、频率及 profiler 干扰限制其精度。此值不是端到端可消除时间，也不是理论下界。

端到端结果必须引用同版本独立配对测量，不能从本表直接推出加速比。该外推不能证明已经达到理论性能。
