CAVIR v0.8：V-aware Router 与骨干共适应的离线算法验证

范围
----
独立v08，不覆盖v07。实现网络输出蒸馏、局部LoRA及受控梯度路径，不是DMD/SiD/GAN、不验证完整视频质量或Kernel性能。
运行时仅原model.blocks索引4/14/24的12heads稀疏，其他层Dense；每当前原始Query独立路由，当前段KV全保留，真实原始KV执行Attention。
主目标确认为原WanDiffusionWrapper的第二输出x0。模型原始网络预测flow，由wrapper按原scheduler转换x0，未将其冒认为epsilon。
四个原生时间步为1000、937.5、833.3333129882812、625；通过原scheduler和配置现场核对，不截断浮点时间步。

模块
----
configs/protocol.json：冻结实验协议和数值容差。
configs/prompt_registry.json：32Train/8Val/16全新Test；保留旧16Train/4Val，明确排除旧暴露Test及历史已采集Prompt。
replay.py / collect.py：完整真实noisy输入、历史latent、原生timestep、context_noise、Teacher x0与轨迹身份；原采样重放采集，不从最终latent逆推。
attention.py：一次原始QKV Attention梯度路径、连续Mask代理VJP和预算阈值导数；保留/未保留页都可向Score传代理梯度，不对零Gate取log。
router.py：真正的K-only128->8 Assignment；6->16->1 LogZ校准器；等容量K/V17->32->1独立Pair评分；128->8内容投影及Pair LayerNorm；不使用R4/Set/Signed。
lora.py：仅指定层Q/K/V/O rank8 alpha8无Dropout；所有基础参数冻结；记录规范化Base哈希及相对有效LoRA更新。
cache.py / backbone.py：函数式KV写入、原位置RoPE、只读版本化Cache与Sketch、完整30层前向、纯checkpoint重算；最近一个历史Chunk的写入梯度保留，更早历史用当前Student参数no_grad重建。
warmstart.py：全新K-only共同校准，不迁移旧Joint-KV S8；仅Train初始统计归一化，Val不更新归一化器。按校准与实际Hard Attention共同选择。
train.py：F/A/J六组、共享warm-start、配对采样、独立Router/LoRA LR与clip、网络x0 Loss、四步一次Dense anchor、渐退Attention辅助与持续cal辅助。
evaluate.py / metrics.py / report.py：每输入NMSE后Prompt等权主指标；Prompt bootstrap；G_F/G_J/I；initial/best/固定last及三预算；完整新Test封存锁。
diagnostics.py：在当前Student自己的QKV上测LogZ/Mass/Value偏差、同Student预测Mass与真值Mass、预算/Gate、可行交换方向；不进入评分前向。
rollout.py：相同原生噪声调用与scheduler、四个3-latent-frame Chunks的Student自生成闭环，不解码VAE；可在Student外部状态上重新求Teacher作为分布移位诊断。
audit.py：CPU数学单测、小型真实Wan模块和完整1.3B/真实输入Phase0审计。
workflow.py：最多两张GPU、每卡一个worker；失败停止后续派发；旧PID不复用；源码变化使审计失效。

数据
----
旧QKV仅供初始校准/辅助诊断，不能用于更新骨干。
原采集代码保存的48个最终latent文件不含逐步noisy状态。已增加真实轨迹采集；每条12-latent-frame轨迹提供3个历史长度×4steps=12个完整Replay输入。
正式数据应为：Train32x2x3x4=768输入、Val8x2x3x4=192输入、全新Test16x2x3x4=384输入。
所有主输入都是完整3 latent frames / 4680当前queries，不把旧128采样queries作为网络输出监督。
8x16个完整Query仅用于抽样辅助项/机制诊断，尾部不足16的query仍参与完整主前向。
上下文重编码沿原context/cache-write的Dense流程；稀疏只用于denoise阶段。Teacher和Student独立构建KV；Student不复用Teacher内部QKV。
原causal推理并未做unconditional/CFG混合；记录配置guidance3.0、实际conditional-only scale1.0，不擅自新增CFG。

执行（在Self-Forcing仓库根目录）
---------------------------------
PY=streaming_sparse/.venv/bin/python
$PY -m streaming_sparse.v08.collect --inventory --registry
$PY -m streaming_sparse.v08.workflow                     # 只生成并显示计划
$PY -m streaming_sparse.v08.workflow --run --through prepare
$PY -m streaming_sparse.v08.workflow --run --through pilot
$PY -m streaming_sparse.v08.workflow --run --through matrix
$PY -m streaming_sparse.v08.workflow --run --through final
$PY -m streaming_sparse.v08.workflow --run --through test
$PY -m streaming_sparse.v08.workflow --status

直观状态面板（只读，不启动实验）
------------------------------
streaming_sparse/v08/status.sh                 # 查看一次
streaming_sparse/v08/status.sh --watch         # 每5秒刷新，Ctrl+C只退出面板
streaming_sparse/v08/status.sh --watch 10      # 每10秒刷新
streaming_sparse/v08/status.sh --no-gpu        # 只读取磁盘元数据
支持从任意目录通过绝对路径调用status.sh。显示Phase0审计是否匹配当前源码、Replay数据覆盖、已启动队列与后续计划、正式6组x3seed进度、运行/失败任务日志及GPU显存和利用率。
数据计数来自manifest，不代表每个tensor已再次校验；源码/梯度正确性以审计报告为准。PID与磁盘状态可能不同步，面板会提示进程退出或不可见，不自动重启。
monitor/为独立的只读工具，不修改科研源文件、审计报告或调度器状态，不加载模型和tensor。

执行加速（2026-10-20）
--------------------
execution/：独立执行优化与等价性/计时审计。Page64选页在GPU上执行相同的稳定降序扫描结果；独立Query评分和连续Mask VJP按128合批，临界排名回退到原16-query计算。原始Hard Attention仍为全4680 Query、12 Heads、原生BF16 SDPA；辅助项仍为同一随机抽样的8x16 Queries。
报告：reports/execution/benchmark.json；早期未通过或较慢的尝试保留在reports/execution/attempts/。完整输入审计通过且源码匹配，launcher才允许启用优化。
启动/继续优化版队列：
$PY -m streaming_sparse.v08.launcher.run --through pilot
单卡继续正式矩阵：$PY -m streaming_sparse.v08.launcher.run --through matrix --gpus 0
--gpus使用物理GPU编号；被排除的卡即使空闲也不会使用。资源选择写入状态，后续省略--gpus时继承；显式--gpus 0 1可恢复双卡。
后续可显式选择matrix/final/test；依赖图、训练步数、Prompt划分、监督和预算不变。调度器已有运行进程时不可重复启动，需先核对status面板。
launcher/记录执行代码、启动代码及审计文件各自的SHA；reports/launched_execution/保留本次代码快照；Checkpoint也保存执行身份。基础科研代码哈希与原Phase0审计保持原样，执行优化使用额外的独立审计，不把基础源码哈希冒充新执行实现的身份。
额外recovery.pt每10个optimizer updates保存一次；不替代按Val选择的best.pt，也不触发额外Val或改变训练样本。其AdamW更新与原实现的完全一致性已做独立检查。
速度审计采用完整14040历史、4680当前Query、36Pair的J-V前向+反向，计时仅代表该固定输入，不承诺全训练吞吐。网络输出NMSE/最大绝对误差仍按protocol原门槛，未放宽；同时比较Router/LoRA参数梯度。

阶段含义
--------
prepare：小型/数学审计，补齐32Train/8Val的两generation seeds真实Replay，再执行完整1.3B与36Pair真实QKV审计。
pilot：seed0共同K-only校准2000步；K/V各200步Head warm-start；四个核心F-K/F-V/J-K/J-V临时小样本完整输入优化检查。
matrix：从正式warm-start重新分叉，四个seed0核心组先执行，再补A-K/A-V；其余两个seed完整六组都执行。所有主组2000updates、每update4完整输入，不按性能选择性追加种子。
final：锁定全部18组的initial/best/last；三预算完整Val及机制/预测Mass参考；汇总配对主效应。
test：仅在锁定后采集与评估新增Test；initial/best/last均评估，同checkpoint三预算；锁定best做四Chunk闭环。
默认--run只执行到pilot，正式18组与独立Test需要显式选择相应through阶段。源代码已提供这些阶段，不代表已经训练完成。
临时pilot在diagnostics/独立目录，不能初始化正式组。H0/R1不排入默认任务；只有锁定主结果稳定时才可使用train --diagnostic H0/R1，R1限制为独立小规模pilot，二者不可同时打开。

梯度和损失
----------
默认仅路由特征副本detach；执行Attention的QKV保留原图。A组冻结整个Router但每次重新评分，F组冻结LoRA仍保留Router所需激活反传。
Lout：公共x0输出输入级NMSE；Lanchor有效0.1，每四updates算一次乘4；仅Router的Latt从0.1在500updates内降0，Lcal有效0.01除以固定Train初始尺度。
正常训练不让辅助统计Loss进入LoRA；R1诊断即使允许主Router输入梯度，辅助项仍使用独立detached副本。
历史latent固定，最近Chunk仅保留其编码写入KV图，不回传生成历史latent的去噪轨迹。Cache/Sketch不能跨optimizer更新复用图；参数版本变化自动失效。
Base/Teacher/模型源码/数据fingerprint、逐参数清单、normalizer摘要、warm-start来源、覆盖率、初始/最优/末尾checkpoint均记录。

完成与限制
----------
当前实现状态见reports/IMPLEMENTATION_STATUS.txt；数学/小型/真实网络审计分别见phase0_unit.json、phase0_real.json。
Phase0通过只证明接口、信息边界和指定梯度的正确性，不证明STE是离散最优梯度，也不证明模型训练有效或在某张卡上收敛。
固定2000updates是研究投入，不是收敛声明。训练效应必须等全部六组×三seed和封存Test实际完成后报告。

执行精度补充：
16-query微批用于评分表示与辅助项；主执行保留原完整12head、全部当前Query的BF16 SDPA布局，避免拆Head改变backend舍入误差。
原始QKV梯度与连续Mask VJP分别由同一原生hard attention和稳定解析式提供，不重复叠加QKV梯度。
初次显存/数值审计的修复记录见reports/NUMERICAL_IMPLEMENTATION_NOTES.txt。
