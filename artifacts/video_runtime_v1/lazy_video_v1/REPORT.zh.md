# 按需回退评分：短视频验收

保持C_s17、Full_Pset_s17、Adaptive φ、B16和10%预算、512交换＋no-op不变。基线为A-only＋Reader两阶段＋逐位等价旧RoPE；仅将两个每次都执行的排除候选向量聚合，改为数值不稳定动作触发时重建。log质量、正数组归一化及g>0规则保持不变。

理论：H12/T72/M16、R16/A32、D16下，两次FP64排除聚合矩阵乘共2×13824×(16²＋32²)×16＝0.566 GFLOPs；相关完整归一化权重共141.56 MB。正常动作不需要这些向量，移除该计算和中间量；触发回退时用正质量重建，不作主导质量直接相减。重建求和次序改变，属于浮点归约差异，不是截断/降阶近似。

## 模块验收

|形状/组合|前/ms|后/ms|加速|最大评分差|动作/no-op|
|---|---:|---:|---:|---:|---|
|kind_a|4.066|3.473|1.171×|1.39e-17|True/True|
|kind_a_p216|2.922|2.331|1.253×|2.22e-16|True/True|
|kind_a_p216_split|2.926|2.192|1.335×|2.22e-16|True/True|
|kind_a_p288|3.451|2.859|1.207×|1.11e-16|True/True|
|kind_a_p360|3.560|2.969|1.199×|5.55e-17|True/True|
|kind_a_p360_split|3.555|2.685|1.324×|5.55e-17|True/True|
|kind_a_p432_split|4.063|3.111|1.306×|1.39e-17|True/True|

split后缀比较A-only与A-only＋split＋lazy，其余只比较lazy开关。已有主导页移出/微小正分稳定性测试通过，见reader_split_v1/lazy_diagnostic。
整条路由以真实保存输入、原生Query布局预热216→288→360→432页：5.288→4.704 ms，1.124×。

## 单视频

city_walk_s701、9 chunks/105帧、768×512、16FPS、360历史页。GPU3顺序运行，每组完整预热后测两轮，再独立profile；VAE单独预热计时，合计不含编译/加载/视频编码。

|版本|生成/s|含VAE/s|
|---|---:|---:|
|baseline|12.750|17.055|
|lazy|12.236|16.548|

生成加速1.042×，含VAE加速1.031×。数值验收：{'latent_equal': False, 'latent_nmse': 0.02261378802359104, 'rgb_equal': False, 'psnr': 28.82484604227092, 'ssim': 0.9558925978839398, 'lpips': 0.03171830446444801, 'window': [21, 101], 'rgb_mae_255': 2.0531040978810142}。

实现：benchmark_hotspots.py新增--lazy-scores；run_lazy_video.py成对测试；profile_lazy_route.py隔离完整路由；评分实现位于scores_packed.py、scores_moments.py和reader_reduce.py。默认LAZY仍为false。未运行15秒或30秒视频。

剩余热点：Reader向量阶段1.446ms，Risk约0.660ms，标量动作评分0.337ms。Reader增大warps到8/16已验证未改善且改变动作，不能简单以减少寄存器数量判断优化成功。下一步应检查向量系数重复读取和归约布局，同时保留已验证的动作语义。
