# E34：Key矩质量修正，冻结R4/E29/Risk

训练数据只用于每个冻结C种子、layer/head的未中心化均值偏移PCA8；共享于同一layer/head的所有页面。原始QKV来自已有缓存，页面身份按有序token IDs/raw K/summary K在轨迹内去重。无神经网络训练、开发选基、rank搜索、二阶系数调节或Test访问。

配置：E29、Variance-only、Full-Mean+Variance、LowRank8+Variance、Exact-Q-Cumulant2、TrueBeta。全部替换所有页的beta；R4近似、Risk、预测局部响应、Medium池不变，M16一次Set/Independent或no-op，真实Q64评价。使用二阶**累积量**mu+variance/2，不是E19的exp Taylor多项式。

```bash
CUDA_VISIBLE_DEVICES=0 python -m local_experiments.e34_key_cumulants.run --seed 17 --basis
CUDA_VISIBLE_DEVICES=1 python -m local_experiments.e34_key_cumulants.run --seed 29 --basis
CUDA_VISIBLE_DEVICES=0 python -m local_experiments.e34_key_cumulants.run --seed 17
CUDA_VISIBLE_DEVICES=1 python -m local_experiments.e34_key_cumulants.run --seed 29
python -m local_experiments.e34_key_cumulants.report
python -m local_experiments.e34_key_cumulants.audit
python -m local_experiments.e34_key_cumulants.finish
```

从仓库根目录、使用既有虚拟环境。依赖E24规范数据与C模型、E28身份索引、E30冻结响应和真值面板缓存。basis与完整run完成后拒绝覆盖。`online_estimates`仅接收Query、摘要Key/stats、BF16偏移坐标和训练基，不接收原始KV/教师。原始Key只用于离线模拟提交时统计及完整方向方差参考。统计构建阶段没有新在线Query扫描。

所有beta通过稳定logsumexp及正质量混合计算；无beta裁剪，巨大质量相对误差通过log指标和有限值计数报告，不删除失败事件。训练基为FP32，32个页面坐标舍入BF16。未测试Kernel或实际延迟。
