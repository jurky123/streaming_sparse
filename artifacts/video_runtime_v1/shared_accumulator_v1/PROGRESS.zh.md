# 精算与补偿共用主累加器

基线为 6135662 的 base，anchor_scale 和 column_mass 均开启。唯一候选保留完整精算和异常回退。以 mx=le−4、den=1、am=exact 初始化补偿循环，末尾直接输出 am/den 加低秩重建，原先的 rho 合并已删除。四代表对应的 16 倍质量仅由初始尺度体现。Q64、BR64、四个 warp、两级流水、双项 BF16 AV、系数顺序和 basis 投影均沿用部署配置。[候选源码](shared.py)

A100 GPU7，432 页真实捕获，43 个精算页。CUDA Graph 预热，交替测 16 对，每个计时含 8 次回放。基线 1.14291 ms，候选 1.17696 ms，配对加速比中位数 0.97101×，16 对均回退。7,077,888 个 BF16 输出中 1,211 个不同，NMSE 为 1.08e−9，最大绝对误差 0.0078125。阶段一到此结束，当前部署继续使用原实现。[原始计时](primary.json)

| P432 指标 | 当前部署 | 共用累加器 |
|---|---:|---:|
| 每线程寄存器 | 255 | 255 |
| 编译器 spill 槽 | 72 | 26 |
| 每 CTA 动态共享内存 | 80 KiB | 80 KiB |
| 实测活跃 warp 占比 | 12.27% | 12.21% |
| local load sectors | 5,529,600 | 4,879,872 |
| local store sectors | 995,328 | 345,600 |
| long scoreboard / active warp | 13.11% | 12.25% |
| short scoreboard / active warp | 13.50% | 13.60% |

硬件计数为 GPU5 单独复核的同形状、预热后一次调用。按编译器行信息把 SASS PC 映射到精算 helper 和补偿源码。基线精算 helper 归属的动态 LDL / STL 指令为 1,161,216 / 169,344，候选为 1,168,128 / 27,648。补偿归属的动态 LDL / STL 为 0 / 0 和 3,456 / 0。另有入口及结束等位置的 local 指令，聚合文件单列为 other。补偿归属的动态 warp 指令从 71.84 M 增至 78.21 M；其中 stall_wait 样本占该阶段样本的比例从 28.7% 升至 35.9%。行映射受指令调度影响，采样适合解释趋势，无法从中唯一确定依赖链。资源节省确实发生，但完整核耗时增加。[分阶段计数](counters.json) · [计数归属脚本](counters.py)

编译产物保存在本目录，包括两版的 TTGIR、PTX、cubin、SASS、资源信息和 Nsight Compute 报告。[部署 TTGIR](baseline.ttgir) · [候选 TTGIR](shared.ttgir) · [部署 SASS](baseline.sass) · [候选 SASS](shared.sass)
