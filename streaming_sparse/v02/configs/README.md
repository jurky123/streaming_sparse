# 配置

- `protocol.json`：固定 Prompt 原文、Train/Val/Test 隔离划分、生成及 Router Seeds、覆盖维度、训练规模。
- `pilot_capture_prompts.txt`：本次 Pilot 采集使用的 Prompt。
- `selected_regularizer.json`：P0 选定的聚类正则权重及选择来源。

每次运行的完整配置保存在 `../runs/{stage}/{variant}/seed*/config.json` 与 `requested_config.json`。
