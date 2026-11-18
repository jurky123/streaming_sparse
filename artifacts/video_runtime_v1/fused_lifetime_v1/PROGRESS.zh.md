# 融合精算与补偿的寄存器实验

当前 P432 主算术约 158 GFLOPs。基线每线程 255 个寄存器、74 个 spill 槽，每 CTA 80 KiB 动态共享内存。Nsight Compute 在单次真实输入上记录 1,022,976 个 local-load sector 和同样多的 local-store sector，各约 32.7 MB；global-load sector 为 55,133,940 个。这是 L1TEX 的请求计数，不能将它视为落到 DRAM 的字节数。长、短 scoreboard 和 barrier 停顿分别占每活跃 warp 周期的 11.61%、15.60%、7.41%。[硬件计数](ncu_local.csv)

以同卡交替 CUDA Graph 计时，去掉 2 对预热、保留 16 对，每次回放 8 遍。四种窗口均复用此前捕获的 layer15/chunk8 输入，较小窗口截取页列表并取模。表中是最常用 P432 的结果，链接包含全部窗口与样本。

| 变体 | P432 加速比 | spill 槽 | BF16 输出逐位相同 |
|---|---:|---:|---|
| [尾部重新读 Q](qreload.json) | 0.9872× | 64 | 是 |
| [逐项消费 detail](late.json) | 0.9802× | 70 | 是 |
| [逐项消费并重读 Q](late_qreload.json) | 0.9775× | 62 | 是 |
| [延迟精算归一化](delaynorm.json) | 0.8750× | 78 | 是 |
| [逐块从 LUT 做掩码](mask.json) | 0.9842× | 70 | 是 |
| [精算流水单级](exact1.json) | 0.8875× | 70 | 是 |
| [补偿流水单级](tail1.json) | 0.9959× | 70 | 是 |
| [补偿流水三级](tail3.json) | 0.8990× | 76 | 是 |
| [全部精算向量写 CTA 共享内存](sharedfull.json) | 0.6602× | 24 | 是 |
| [一半精算向量写 CTA 共享内存](sharedhalf.json) | 0.6642× | 56 | 是 |
| [每 CTA 处理 32 行](rows32.json) | 0.6623× | 0 | 是 |
| [每 CTA 两个 warp](warps2.json) | 0.4473× | 448 | 否 |
| [补偿块扩大至 128 代表键](br128.json) | 0.5764× | 198 | 否 |
| [128 代表键并重读 Q](br128_qreload.json) | 0.5745× | 198 | 否 |
| [两个精算页交错做 QK](pair_inter.json) | 0.3782× | 244 | 否 |
| [两个精算页连续做 QK](pair_contig.json) | 0.2175× | 368 | 否 |

32 查询行方案的 spill 降到 0，但 CTA 数翻倍且每个 CTA 重新加载所选页，速度回退。把精算向量手动留在共享内存也降低了 spill，同时增加每 CTA 的共享内存需求。上述变体均未进入默认部署。
