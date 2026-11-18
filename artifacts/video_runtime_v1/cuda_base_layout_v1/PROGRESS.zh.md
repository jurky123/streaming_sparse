# 固定 15% base 切换 CUDA，clean 提交直接写入矩阵片段布局

base 在 SM80 的默认路径采用 Q64、双项 BF16 AV、首帧 24 页计入固定 15% 总预算。每个稀疏层有一份额外的 packed K/V，全模型共 4.588 GiB。261 次 clean 提交在写入原生 K/V 的同时产生该布局；812 次稀疏调用只需重排当步 72 页。intersection 与 step 继续使用各自的 10% 精算与 correction 策略。CLI 通过 --no-cuda-fixed15 复现前一版 base；常驻 worker 按作业选择实现。

## 三种模式的同卡速度

| 模式 | 生成中位数 s | 相对同卡 Dense | 相对 Dense 的 latent NMSE |
|---|---:|---:|---:|
| base | 5.28041 | 1.237× | 0.557547 |
| intersection | 5.20506 | 1.255× | 0.379339 |
| step | 5.91565 | 1.104× | 0.347492 |
| 原生 Dense | 6.53286 | 1.000× | 0 |

完整热身一次后计时五次。三个模式的最终 latent 各自与前一版逐位相同。

## 直接生成布局

在真实 P432 输入中取 72 页并组成 clean 分组的 head-major 输入。原实现写原生缓存后重排，耗时 141.95 µs。直接 scatter 写两种格式耗时 143.55 µs。新 CUDA CTA 先将一个 head 的 16 行写入共享内存，再以 uint4 对齐写出 MMA 操作数，耗时 **81.22 µs，1.748×**。逐槽比对包含首帧、历史环绕、reset 后复用和全部未写槽位。连续 20 次提交的 K/V、打包片段、摘要、token ID、写入位置均与原实现一致。

同一 city_walk / 701 视频做 6 对正式入口交替，单独启用直接提交的配对中位收益 **12.20 ms**，6/6 更快。原型单视频有一次负收益，因此按约定做了 5 prompt × 2 seed，9/10 更快，配对中位节省 **10.69 ms**，范围 −2.12 至 17.46 ms。10 组所有 chunk 的 latent 和计数一致。整项改动（CUDA 融合与直接提交）和旧 Triton base 交替 6 对，6/6 更快，配对中位节省 **52.60 ms**，所有 latent 逐位一致。release_pairs、triton_pairs 记录正式入口，paired_video、multi_video 记录直接提交原型。

当前页的投影出口另作模块试验。原 RoPE 后单独重排耗时 90.24 µs，在线程已有的 K 结果上分散写两种布局并同时搬运 V 需要 200.51 µs。分离 RMS 归约再以 16 行合作写出后为 94.53 µs，同轮对照 90.75 µs。两种均逐位一致，当前页继续单独重排。

## clean 全注意力

输入捕获自 city_walk / 701、chunk 5、layer 1、clean step 4、P432。三种实现保持全部页参与 softmax，计时使用预排好的 K/V。

| clean 完整模块 | 中位 ms | 相对原生模块输出 NMSE |
|---|---:|---:|
| 原生 Dense SDPA | 3.7551 | 0 |
| 手写 SM80，直接消费 packed K/V | 4.0465 | 1.13e-6 |
| Triton，直接消费 packed K/V | 14.1823 | 1.13e-6 |

另测以首个页为尺度的 CUDA 原型，4.0845 ms，864 个 tile 有 5 个触发质量标志，此时间尚未计入重算。以上 clean 原型均止于模块测试。当前 clean 保留原生 Dense。

## 设备时间线

105 帧主核累计 869.90 ms，当前 K/V 重排 39.75 ms，摘要重排 12.26 ms，guard 12.57 ms，稀疏注意力共 934.48 ms。原 CUDA 的 K/V 重排为 51.56 ms。新的 clean 双布局写入在提交阶段累计 18.92 ms，其工作包含原本已有的原生 KV 写入。墙钟净收益使用上列配对。按 150.289 TFLOP 计，主核为 172.76 TFLOP/s，稀疏注意力路径为 160.83 TFLOP/s，距逐窗 Dense 时间参照 726.34 ms 仍差 208.14 ms。

## 复现与证据

三个模式使用同一张 A100-SXM4-40GB，city_walk / 701，512×768，105 帧。每个模式先完整预热一次，然后计时五次，记录生成墙钟（含 clean 提交）。运行 python -m local_experiments.video_runtime_v1.accepted_tiers --tier MODE --reps 5 --skip-profile --output FRESH_PATH --local-assets /tmp/streaming_sparse-0/runtime。release.json 保存逐次计时、计数及 Dense 误差。四窗真实输入与质量极端输入见 guard_*.json。

生产源码为 local_experiments/video_runtime_v1/csrc/fixed15_attention.cu、fixed15_attention.py 和 fixed15_layout.py。release_pairs.py 复现配对，--triton-control 对照旧 base。capture_clean.py 提取 clean 输入。独立原型用 nvcc -arch=sm_80 -O3 --fmad=false -shared -Xcompiler=-fPIC 编译，脚本按其中指定的 /tmp 库名加载。模块测量使用各窗捕获文件 /tmp/jzk_fixed15_input_P.pt，20 组交替 CUDA Graph 计时，每组重放 8 次。
