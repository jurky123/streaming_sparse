# P2：Reader 小 WORK 有界实验（结束，不部署）

外层始终 tile72，12 heads × 72 tiles × 16 probes = 13,824行、432页。沿用9d4d33d全部已验收P1配置，scan_fp32关闭，只改变Reader内部scalar/vector交替生产消费的行数。GPU5，同一捕获事件，预热后CUDA Graph 50次中位数。

|内部行数|WORK/MB|scalar/vector对数|Reader/ms|Reader加速比|完整路由/ms|路由加速比|
|---|---:|---:|---:|---:|---:|---:|
|全部|283.116|1|1.9046|1.0000×|4.3377|1.0000×|
|512|10.486|27|2.1164|0.9000×|4.4980|0.9644×|
|1024|20.972|14|1.9910|0.9566×|4.3752|0.9914×|

两种分批的3个Reader输出、5个路由输出均与原实现逐位一致，动作不变。512和1024都没有净收益，因此停止P2，不继续warps/寄存器/粒度搜索，不启动P3，也不对失败候选跑视频。

理论变化是WORK峰值容量缩小27×/13.5×，逻辑写读量仍为566.231MB；小工作集可能提高缓存命中，但增加26/13对kernel发射。实测Reader反而分别增加0.212/0.086ms，不能把缓存容量缩小直接当作流量或延迟同比减少。没有硬件计数器证据把回退单独归因于发射或缓存。

## 实现与复现

- `local_experiments/video_runtime_v1/reader_chunked.py`：仅增加运行时ROW0，全局输入输出索引不变，WORK使用局部行索引；逐行数学、页分片和归约保持不变。
- `reader_reduce.py`：显式work_rows参数和fake签名，循环交替scalar/vector，默认WORK_ROWS=0。
- `check_p2.py`：固定0/512/1024三项、同输入输出规约、计时与profile。
- 原始证据：[result.json](result.json)、rows0/512/1024_profile.json及txt。

复现：在仓库根运行 `CUDA_VISIBLE_DEVICES=5 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.video_runtime_v1.check_p2`。脚本写同名结果目录，需先另存已有结果。
