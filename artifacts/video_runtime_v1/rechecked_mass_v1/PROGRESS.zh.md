# 分块计算路由 QK 与 Risk

P432 下，12 个头、1152 个 probe、1728 个代表键的 QK 约为 6.12 GFLOP，写出 FP32 logits 需 91.1 MiB。原路径先写 logits，再分别为 softmax 行归一化和 Risk 页归约读取。上轮整段时间线中，行归一化约 66 ms。

## 模块筛查

30 轮交替计时，每轮 10 次 CUDA Graph 回放。P432 下，QK、行归一化和 Risk 共约 0.352 ms。改变 logits 内存布局、按头分批或循环复用输出，最好约持平。cuBLAS 替换后约 0.359 ms。单 CTA 完整计算 16 个 probe 的所有 logits 与行归一化约 0.426 ms，并改变归一化分母。FP16 logits 约 0.309 ms，但真实输入 864 个查询块中有 9 个选页集合改变、263 个页序列改变。该变体停留在模块阶段。[布局](../logit_layout_v1/module.json) · [复用](../logit_layout_v1/reused_heads.json) · [整行融合](../logit_layout_v1/full_rows.json) · [FP16](../logit_layout_v1/fp16.json)

## FP32 分块与误差来源

对同一 probe，四项 Risk 特征都是 softmax 页内权重的一次齐次函数。先在 128 页块内计算 QK、局部最大值、指数质量及逐页贡献，再根据各块质量还原整行加权和。主要中间量从每页四个 FP32 logits 变为一个 FP32 贡献，另存每块两个 FP32 归一化量。432 页补齐成四个 128 页块，两个核合计写约 27.0 MiB 贡献和 0.42 MiB 块统计。初版在四个窗口加速 1.23–1.43×，模块 fixture 的页序列相等。分块重新安排浮点运算顺序，会改变极为接近的 Risk 排序。[模块](../logit_layout_v1/partial_mass_wide_w4.json)

city_walk/701 初版的 base 和 step 最终 latent 相同，intersection 整段 NMSE 为 0.09656、最大绝对差为 4.359375。沿 base 旧轨迹逐次核对 701568 个查询块，有 20 个页序列变化。该初版关闭。[初版单视频](../partial_mass_video_v1/intersection.json) · [轨迹核对](../partial_mass_video_v1/trace.json)

## 接近排序边界时重算

对强制保留的本页以外，检查前 K 名中的相邻分数和第 K/K+1 名的差距。相对间隔小于等于 4e-5 的查询块重走原 FP32 QK、行归一化和 Risk 归约，然后再选页。阈值根据旧轨迹中约 1.27e-5 的最大相对 Risk 差异选取，是经验阈值。校验使用包括帧块 2–8 在内的当前部署第 15 层输入。七个事件的对照 Risk 和页序列与实录逐项相同，候选恢复了全部被标记查询的原分数。部署只在无需 Full_Pset 纠正的调用中使用这条路径，纠正分支继续读取原 FP32 logits。

逐次比较 base 的 701568 次和 intersection 的 659232 次适用选页，集合与顺序全部相同。早期检测原型使用浮点排序的无穷填充时标记错误，已改为对非负 Risk 的无符号位编码排序，并按 torch.topk 查验排序及标记。记录使用修正后的实现。[重算模块](../logit_layout_v1/recheck_events.json) · [base 轨迹](../partial_mass_video_v1/recheck_base.json) · [intersection 轨迹](../partial_mass_video_v1/recheck_intersection.json)

## 单视频

city_walk，种子 701，105 帧，仅计生成，每档交替 10 对。所有配对的 latent 逐字节一致，路由计数相等。

| 档位 | 旧中位/s | 新中位/s | 加速 | 配对节省/ms |
|---|---:|---:|---:|---:|
| [base](base.json) | 5.38550 | 5.35513 | 1.0057× | 18.0–43.5 |
| [intersection](intersection.json) | 5.45734 | 5.42810 | 1.0054× | 12.9–43.6 |
| [step](step.json) | 6.11245 | 6.08729 | 1.0041× | 9.1–38.0 |

默认入口完整预热后各跑五次。base、intersection、step 中位数为 5.32914、5.41137、6.08224 秒，三档最终 latent 与上次部署字节一致。沿用 GPU7 同主干原生 Dense 的 6.58614 秒，base 加速 1.23587×，距 4.70324 秒还差 0.62590 秒。整段 profile 中，分块 Risk 的第一阶段 164 ms，第二阶段 29 ms，精算 534 ms，补偿 358 ms。[正式入口](deployment.json) · [时间线](device_timeline.json)
