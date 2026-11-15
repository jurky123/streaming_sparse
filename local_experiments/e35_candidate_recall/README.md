# E35-A：候选动作覆盖重放

只读取E30保留的E29初始512动作预测分数、真实Q64收益与预测响应缓存。不加载原始QKV，不运行Attention/Reader，不替换beta，不执行E35-B。

```bash
python -m local_experiments.e35_candidate_recall.run
python -m local_experiments.e35_candidate_recall.audit
python -m local_experiments.e35_candidate_recall.finish
```

L=1/2/4/8，按预测分数降序，平局按原面板下标。排序不看真值，不用预测正收益阈值筛候选。每个Top-L的收益上限由真实标签取最佳正收益，含no-op。它不是可部署决策结果。

额外报告Set/Independent各前2提议的动作并集、以及所触及页面构成的合法笛卡尔交换池；后者才对应建议B的候选空间。均保留Medium分母，不将受限池Oracle与Medium混用。Top8动作不等于8页。

Independent仅由冻结预测响应/beta/mass/center重算页面残差能量顺序。E30的Set分数为离线单位对齐除以Dense正能量常数，但该正常数不改变排名。主基线Top1重现既有事后真门控结果。

两个模型种子、相同两开发轨迹、各288相关head–tile；不是576独立视频。预登记参考门槛70%收益/80%正动作覆盖，尚非在线成功标准；无论门槛如何，本轮只完成A。
