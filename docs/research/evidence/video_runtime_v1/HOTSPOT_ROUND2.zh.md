# 交换评分融合优化

在 GPU6 对单个 city_walk 捕获事件扩展的 72 个 head–tile 工作量进行模块隔离测量；GPU7 执行数值正确性验证。使用 torch.compile(fullgraph=True) 融合 FP64 逐元素计算及归约，内部 cudagraphs 关闭，由现有 BatchGraphs 管理捕获与预热。保留 eager grouped_scores 作为参考。运行时 graph/eager 两个入口均接入 fused_grouped_scores。

15 次预热后测量：墙钟中位数 4.929 → 1.350 ms（3.65×）；额外峰值分配 491,905,536 → 22,735,872 bytes。独立 CUDA Graph replay 25 次中位 GPU 时间 4.103 → 1.090 ms（3.77×），确认收益并非只来自 Python 开销。

14 个 E49 案例通过，涵盖微小正收益、严格 no-op、动作 argmax；主导页相差 1000 log 单位的移出测试通过。单视频性能输入最大评分差异 5.55e-17，动作一致。完整验证的 Attention、Reader、Risk 门禁也通过。

测量不含首次编译时间，新形状需要首次编译及 graph 预热；这些是模块结果，尚不能推断完整视频同倍率加速。训练 GPU0–3 入口未改动。原始结果位于 artifacts/video_runtime_v1/hotspot_round2。
