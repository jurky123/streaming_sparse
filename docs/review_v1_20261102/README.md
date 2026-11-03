# 2026-11-02 评审方案实施记录

本轮遵循：先验证质量，再专门优化速度。旧的 `GapGainVector64Runtime` 保持不变。新代码位于 `local_experiments/review_v1/`，目前是独立质量参考实现，不能用于宣称加速兑现。

## 已实现

- 配置统一历史容量338页、当前72页、page64、16 probes、4代表、rank8，默认固定10%预算；支持整除容量时 drop=0。
- 完整 BF16 V 均值 + 零和 Haar 细节；保留有界可学习 K/V 配对池化、K 页均值重居中。移除旧中心/低秩基值语义。
- 基于部署量化后的 K/V 重建值计算8组 K 方差、bK/bV/sV/bV0/Bp，结构分离，1,424字节/页/头。
- 两遍归一化残差风险参考、每层每头4个非负系数、受保护当前页、动态计数/溢出/剩余风险接口。
- 页面流式实际残差标签，避免存储全尺寸残差向量或原始QKV。
- 显式 PageStore / GroupingOps / SummaryOps / RouteOps / SelectedAttention / StreamingRuntime；历史仅在 clean 提交，当前每去噪步生成；层0、前两chunk、clean仍Dense。
- 精确分支只取实际被选页，未选页用共享归一化的完整均值/细节补偿。当前 PyTorch 参考动态计数会主机同步，QK/AV为FP32参考（输入先BF16量化），未实现生产TF32数值路径，也未进行速度调优。

## 验证与阶段门槛

CPU契约检查：`CUDA_VISIBLE_DEVICES='' bash local_setup/run.sh -m local_experiments.review_v1.test_ops`。

真实输入对照入口：`gap_review_v1_probe`。先在旧方案轨迹的同一KV状态对照旧摘要/旧mask、新摘要/旧mask、新摘要/初始风险mask。层1/15/29，chunk2/5，step0，每头两个tile各16真实query。输出仅标量结果及小型路由校准特征。旧方案结果是代数参考，不冒充原kernel数值结果。这个检查不等于视频质量验收。

后续门槛：

1. 固定mask训练摘要，骨干冻结；池化、基、组特征均参与。硬排序无虚构梯度。
2. 摘要冻结后，用实际残差标签校准非负theta；训练/验证视频分离，摘要变更使标签与阈值失效。
3. 按层/head/step校准lambda，以总选页数/总候选页数校准10%，不是各形状密度简单平均；warmup/full/eviction都要覆盖。
4. 学生状态两步下游蒸馏与早/中/后段30秒视频质量验证。当前局部检查不代替这两项。
5. 质量确认后再实现设备counts循环、两遍路由融合、TF32补偿、共享arena/图回放，并测真实端到端速度与峰值内存。

本轮不默认启用8%/6%、sparse clean、扩大历史、减少probe或终端clean跳过。暂不宣称≤256MiB瞬时工作区、完整训练收敛、视频质量提升或端到端加速。

## 运行与存储

实验根目录 `artifacts/review_v1_20261102/`；每事件原子写入，已完成事件重启跳过，按case重放恢复缓存，不保存大QKV。磁盘空闲下限50GiB，单组输出上限1GiB。GPU7检查发现其他用户进程时等待，不停止对方任务。

## 本次检查记录

7项CPU数值/梯度/选择/校准测试通过。按CPU加载的实际模型参数核算：原始KV 4,845,207,552 B；摘要167,496,576 B；token身份60,223,488 B；稀疏参数14,482,368 B；合计5,087,409,984 B，小于5,112,856,576 B。此值未计瞬时工作区、骨干、CUDA图池，不能当作GPU峰值显存。

GPU预检与初始真实输入对照已完成，摘要训练已启动。原CPU等待器已停止；因OS拒绝停止其他用户进程，改为独立子进程的共享GPU质量执行器 `gap_review_v1_worker`，仅重启自己的会话。状态文件：`artifacts/review_v1_20261102/worker/status.json`。详见 [pilot进展](pilot_progress.md) 与 [评审核对](preflight_audit.md)。

自动队列包含：鸟类初始同状态对照；city_walk固定mask摘要训练pilot（3层、2个chunk、每事件4步）；鸟类留出复测；city_walk冻结摘要残差采样；非负theta拟合；鸟类路由复测。训练先使用局部attention输出目标验证模块，完整29层两步下游训练及30秒闭环视频仍待pilot结果后推进。`gap_review_v1_train --all-layers` 支持29层摘要预训练，但不是完整两步下游训练。

当前执行器重启命令：`CUDA_VISIBLE_DEVICES=7 GAP_TAIL_MODE=1 bash local_setup/run.sh -u -m local_experiments.gap_review_v1_worker`，先确认没有另一实例。旧CPU等待器不要同时启动。结果按事件/任务原子保存，故障停止后可检查状态再恢复；不会自动略过数值失败。

最新状态：首轮pilot全部完成，详见 [完整pilot结果及校准缺口](pilot_results.md)。目前等待分析后的下一轮处理，不再处于训练运行状态。

第二轮正在推进：[校准修正、扩大训练与验收计划](round2_design.md)。四系数重新拟合虽收敛但选页退化，已保留结果并改用保持页排序的尺度校准候选。
