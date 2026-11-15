# E32 Candidate-only Key Probing

冻结 E24-C、E27 adaptive phi、E29 local response Reader；两模型种子各288开发事件。只探测初始 Medium 的48页，M16，每16-token子组 b=1/2/4，单次交换或 no-op。无训练、新QKV、Test或Kernel。

## 复现

使用仓库现有实验虚拟环境，从仓库根目录执行：

```bash
python -m local_experiments.e32_candidate_key_probe.run --plan
CUDA_VISIBLE_DEVICES=0 python -m local_experiments.e32_candidate_key_probe.run --seed 17
CUDA_VISIBLE_DEVICES=1 python -m local_experiments.e32_candidate_key_probe.run --seed 29
python -m local_experiments.e32_candidate_key_probe.report
python -m local_experiments.e32_candidate_key_probe.audit
python -m local_experiments.e32_candidate_key_probe.finish
```

依赖已有 E24 规范QKV、E29冻结模型、E30 beta与共同动作面板缓存。计划首次生成后冻结且记录SHA；完整run完成后拒绝覆盖。所有实验参数在 `artifacts/e32_candidate_key_probe_v1/protocol.json`。

采样种子17/29/43/71/101是独立重复试验，不是线上五次探测集成。采样由trajectory/layer/head/ordered token IDs及采样种子确定，对同一物理页的各上下文及两个模型种子保持一致，b嵌套。决策只读取候选页抽中的K。离线重建冻结摘要/真值时加载的完整KV不进入 `probe_beta` 函数。

Set沿用E30数值稳定的正质量混合目标，不裁剪beta。Independent使用完全相同页面信息的独立残差能量排序，在同池选择一次交换或no-op。教师仅用于对照/评价。真实Dense能量只用于正尺度对齐离线分数，不影响在线动作排序或no-op。

原始运行文件保留每事件候选beta、初始/五状态共同面板预测与真值；报告CSV和精简tar不包含规范原始QKV或模型权重。成本只报告新增Key探测部分，未测实际gather/kernel延迟，不将统计重复次数算作部署成本。
