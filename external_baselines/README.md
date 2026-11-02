# 固定版本的外部基线

`sources.lock.json` 记录官方公开仓库与commit；`patches/` 保存本轮使用的本地兼容改动。
外部仓库本身不作为嵌套Git仓库上传，各项目原始许可证保留在克隆目录。

在仓库根目录运行：

```bash
python3 local_setup/fetch_stream_baselines.py
```

脚本不会覆盖现有目录或重置已有改动；仅接受固定版本且补丁可干净应用或已应用的情况。
模型资产必须先恢复到主仓库 `wan_models/`、`checkpoints/`。
DASA兼容补丁修正有限缓存溢出和压缩指针；LightForcing补丁修正Triton归约类型兼容。
这些是带兼容补丁的官方完整推理管线，不能称为未经修改的论文代码测速。
