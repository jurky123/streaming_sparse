# E33-A：四代表页面质量响应容量

只做页面参数的离线教师拟合，不训练网络或修改正式推理实现。固定 E24-C 两种子，开发集两轨迹×3层×head0×3chunk×2step×2tile=72事件/种子。当前页面及没有合法 donor 的历史页始终保持原摘要。三个留出条件分别为其他 tile、其他 step、严格过去 chunk，另有共同支持集对照。

A1 固定 K/V，仅优化归一化正权重。A2 固定等权和 V，只优化四个 Key，解除原 pooling 凸组合/均值约束，因此是更宽的表示容量诊断，不等同于原 pooling 网络可以生成这些 Key。两者用 donor logZ 拟合200步，未看留出质量调超参数。评价使用 BF16 rounded Key/weight，权重重归一化。无权重缓存/Kernel部署或旧Reader迁移评价。

```bash
CUDA_VISIBLE_DEVICES=0 python -m local_experiments.e33_mass_representatives.run --seed 17
CUDA_VISIBLE_DEVICES=1 python -m local_experiments.e33_mass_representatives.run --seed 29
python -m local_experiments.e33_mass_representatives.report
python -m local_experiments.e33_mass_representatives.audit
python -m local_experiments.e33_mass_representatives.finish
```

从仓库根目录，使用已有虚拟环境。依赖E24规范QKV/C模型及E28页面身份索引。输出 `artifacts/e33_mass_representatives_v1/`。完整run完成后拒绝覆盖。精简包不含原始QKV与所有拟合表示，完整事件文件保留拟合表示、支持集、donor映射、新Risk集合及512真实交换标签。

Risk theta不变，四项公式不变；新权重进入分子/分母及路由概率。A2仅更新受影响页面的Key代表偏差统计，均值/Value detail和其他stats不变。另报固定旧Risk集合的输出，以区分摘要与硬路由变化。主基线是对应冻结C，不是原始178f95a8摘要。
