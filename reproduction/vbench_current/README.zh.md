# 当前五方法 VBench 队列

本目录对应 2026-11-19 的 5,760 条生成任务。短视频每档 384 个 prompt × 2 个 seed，长视频每档 128 个 prompt × 3 个 seed。jobs.tsv 固定 rowid、方法、输入和优先级。生成 768×512、16 fps，以原生 VAE 截取 80 或 480 帧。评分共 20,890 项。

worker.py 是五档的实际入口。Dense 使用优化主干的原生滑窗，LightForcing 加载上游长短配置，A100 使用 BF16、RoPE 和 scale-shift 优化。base 和 step 使用部署预设。intersection 另接原始权重、首帧固定 24 页和 10% 非锚点平均精算预算。policies.npz 中的 10_window_step_layer_head 分配当前和历史页，从第七个 chunk 开始相对当前查询重新旋转首帧 K 并刷新摘要。quota_pool.py 修正候选槽重叠。仅调用 accepted_tiers 的通用 intersection 入口会得到不同的视频。

## 新机器

原环境为 Linux x86-64、Python 3.10、8 张 A100-SXM4-40GB。需 NVIDIA 驱动、SM80 CUDA 编译器（生成扩展使用 /usr/local/cuda/bin/nvcc）、C++ 编译工具、Git、curl、ripgrep 和 FFmpeg。生成使用 Torch 2.6.0+cu124 和 Triton 3.2.0；评分单独使用 Torch 2.1.1/cu121，编译 Detectron2 时选择兼容 CUDA 12.x 的工具链。以下命令在克隆的仓库根目录执行。运行目录放在本地快盘，选一个全新的路径。

~~~bash
export VBENCH_RUN=/fastdisk/vbench-current
export PYTHONPATH="$PWD"
code=reproduction/vbench_current
bash "$code/gen_env.sh"
source .venv-vbench-gen/bin/activate
python migration/download_models.py --full
python local_setup/fetch_stream_baselines.py
bash "$code/cutlass.sh"
python "$code/prepare.py"
CUDA_VISIBLE_DEVICES=0 python "$code/embed.py"
python "$code/score_assets.py"
bash "$code/score_env.sh"
python -u "$code/launch.py"
~~~

prepare.py 校验六份项目权重及输入文件，初始化独立 SQLite。公共 Wan、Self Forcing EMA、VAE、T5/tokenizer 由 migration/download_models.py 按 revision 和 SHA-256 取得。LightForcing、Forcing-KV 的 revision 和补丁在 external_baselines 下。score_assets.py 取得本轮实际 VBench 源码、三处本地评分改动、DINO 和评分权重，并校验下载文件。可在下载前将已有的相同权重放到对应路径。VBENCH_SCORE_PYTHON 可指定已建好的评分解释器。生成和评分包版本清单分别为 generation.versions 与 scoring.versions，评分额外加载的包在 score_env.sh 中。

## 核验

运行 python reproduction/vbench_current/status.py 查看进度。生成目录为 $VBENCH_RUN/videos/方法/时长/编号_seed.mp4；生成、VAE 用时分列存于 jobs.sqlite。8 个生成 worker 完成后自动切换为 8 个评分分片，原始结果在 metrics 目录。单卡可运行 CUDA_VISIBLE_DEVICES=0 python reproduction/vbench_current/worker.py intersection long，处理该分组所有待办。每个运行目录使用一个全队列调度器。

东京 long_0/701 和猛犸 long_1/702 在 worker 中自动核验完整 BF16 latent 的 SHA-256，参照来自通过跨视频顺序检查的修正实验。该核验对 GPU 架构、算子版本和权重敏感。MP4 编码器版本可能改变文件字节。history 目录保留预算标定、重定位、微调和速度对照的原实验源码及记录，里面的绝对路径属于旧实验目录。当前推理和评分使用上面的入口。
