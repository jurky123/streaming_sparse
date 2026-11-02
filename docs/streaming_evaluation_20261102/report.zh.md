# 流式视频方法同模型比较

统一512×768、480帧/16FPS、41个3-latent-frame chunk、最多5个历史chunk。相同Self-Forcing EMA骨干、输入噪声和文本嵌入，统一原生VAE。每种完整预热一次后计时两轮；加载和首次编译不计入，输出编码与质量计算不计入生成＋VAE。GPU7串行，计算和图形上下文监控排除经过核实的空闲父进程显存预留。

当前完成 15/15 个场景-方法单元。未完成项不作为失败或零分。

|场景|方法|生成＋VAE/s|生成/s|相对同场景Dense加速|前段LPIPS↓|前段SSIM↑|前段PSNR↑|
|---|---|---:|---:|---:|---:|---:|---:|
|sailboat|dense|38.897|28.229|1.000×|-|-|-|
|sailboat|ours|30.863|20.540|1.260×|0.2177|0.6530|18.0447|
|sailboat|lightforcing|31.699|21.389|1.227×|0.1608|0.6252|18.3086|
|sailboat|forcingkv|28.802|18.352|1.351×|0.1807|0.6143|17.7449|
|sailboat|dasa|32.783|22.374|1.186×|0.1724|0.6675|18.2469|
|skateboard|dense|38.859|28.183|1.000×|-|-|-|
|skateboard|ours|30.495|20.082|1.274×|0.2758|0.4760|15.0092|
|skateboard|lightforcing|32.027|21.451|1.213×|0.2085|0.4928|16.5221|
|skateboard|forcingkv|28.562|18.169|1.361×|0.3600|0.3728|13.5607|
|skateboard|dasa|34.090|23.730|1.140×|0.2321|0.4591|15.8517|
|dog|dense|38.937|28.268|1.000×|-|-|-|
|dog|ours|30.613|20.206|1.272×|0.4292|0.3615|10.1710|
|dog|lightforcing|31.632|21.180|1.231×|0.4225|0.3453|10.4280|
|dog|forcingkv|28.944|18.354|1.345×|0.4255|0.3667|10.0558|
|dog|dasa|36.118|25.717|1.078×|0.3518|0.4501|11.7859|

前段质量窗口RGB[21,101)，PSNR全分辨率、LPIPS/SSIM在256×384计算，均基于未压缩RGB。Dense一致性指标不是绝对质量排名。后续前段DINO/运动/亮度指标作为辅助，不能单独以一致性高或运动低判定质量好。只覆盖三个场景、一个seed，不能外推论文总体质量。

## 配置与方法限制

- Forcing-KV native smaller per-head context retained and reported, not padded to 5 chunks.
- DASA 15 main-cache frames plus 3 hidden clean KV frames within common self-KV cap; auxiliary feature reuse reported separately.
- Our summary metadata allows 16MiB above original raw KV allocation; raw history338 pages vs dense360 pages.
- LightForcing same-backbone weight transfer excludes its paper-specific fine-tuning; native relative position policy remains.
- Dense uses common PyTorch SDPA; official LightForcing sparse Triton path.
- Official bounded-buffer compatibility patches retained and archived; no user rebase modification.

## 三场景汇总

加速比取逐场景比值的几何平均；质量指标为三个场景前段指标的算术平均。

|方法|平均生成＋VAE/s|几何平均加速|前段LPIPS↓|前段SSIM↑|前段PSNR↑|
|---|---:|---:|---:|---:|---:|
|dense|38.897|1.000×|-|-|-|
|ours|30.657|1.269×|0.3076|0.4969|14.4083|
|lightforcing|31.786|1.224×|0.2639|0.4878|15.0862|
|forcingkv|28.769|1.352×|0.3221|0.4513|13.7872|
|dasa|34.330|1.134×|0.2521|0.5256|15.2948|

## 实际内存

|场景|方法|自注意力KV或我方KV+辅助/MiB|官方额外辅助/MiB|
|---|---|---:|---:|
|sailboat|dense|4860.00|0.00|
|sailboat|ours|4875.96|0.00|
|sailboat|lightforcing|4860.00|0.00|
|sailboat|forcingkv|2081.25|0.00|
|sailboat|dasa|4860.00|138.16|
|skateboard|dense|4860.00|0.00|
|skateboard|ours|4875.96|0.00|
|skateboard|lightforcing|4860.00|0.00|
|skateboard|forcingkv|2081.25|0.00|
|skateboard|dasa|4860.00|138.16|
|dog|dense|4860.00|0.00|
|dog|ours|4875.96|0.00|
|dog|lightforcing|4860.00|0.00|
|dog|forcingkv|2081.25|0.00|
|dog|dasa|4860.00|138.16|

我方列出KV与摘要合计，官方方法列出KV及额外辅助两列；官方warm_audit.json还记录预热时峰值。进程峰值显存包含模型/工作区；父进程占卡预留不属于模型部署内存。

各方法视频在本机 `artifacts/stream_methods_20261102/<场景>/<方法>/video.mp4`，未包含在本次源码发布中。全部结果原子落盘，已完成方法可直接跳过。

## 前段独立视频描述指标

这些辅助指标不能取代视频判断；DINO一致性高也可能来自运动停滞。

|场景|方法|DINO相邻余弦↑|DINO首帧余弦↑|运动L1|最大暗闪/百分点↓|
|---|---|---:|---:|---:|---:|
|sailboat|dense|0.9702|0.9302|0.0231|0.885|
|sailboat|ours|0.9588|0.8638|0.0233|0.244|
|sailboat|lightforcing|0.9760|0.9038|0.0260|0.212|
|sailboat|forcingkv|0.9737|0.8811|0.0261|0.149|
|sailboat|dasa|0.9828|0.9458|0.0212|0.183|
|skateboard|dense|0.9617|0.9211|0.0318|0.175|
|skateboard|ours|0.9286|0.8673|0.0322|0.930|
|skateboard|lightforcing|0.9494|0.9055|0.0343|0.232|
|skateboard|forcingkv|0.9434|0.8897|0.0329|0.344|
|skateboard|dasa|0.9548|0.8872|0.0303|0.250|
|dog|dense|0.9151|0.7365|0.0887|0.997|
|dog|ours|0.9000|0.6744|0.0772|0.757|
|dog|lightforcing|0.9165|0.7943|0.0964|2.185|
|dog|forcingkv|0.9247|0.8335|0.0861|1.544|
|dog|dasa|0.9133|0.7913|0.0866|0.794|

## 结论与解读

本轮三个场景的生成＋VAE平均：我方30.657秒，LightForcing31.786秒，DASA34.330秒，Forcing-KV28.769秒，Dense38.897秒。我方相对Dense几何平均1.269×，比LightForcing平均耗时低3.55%，比DASA低10.70%，但比Forcing-KV高6.56%。每种仅两轮，且不同方法在各自子进程内连续测量；这支持首轮描述性比较，不提供小差异的统计显著性结论。

我方平均LPIPS=0.3076、SSIM=0.4969；LightForcing为0.2639、0.4878；DASA为0.2521、0.5256；Forcing-KV为0.3221、0.4513。DASA在这些相对Dense的均值指标最好，我方相对LightForcing的SSIM略高，但LPIPS较差，不能声称质量全面领先。

独立前段DINO指标也没有支持我方更好的一致性：我方在三个场景的相邻余弦均低于LightForcing。指标仍不是绝对画质或文本对齐评价；静止或小运动可获得较高一致性。

人工检查2.5秒和5秒的同屏截图：滑板场景中我方5秒的人物尺度更大、头部超出画面，和Dense及LightForcing构图不同；狗场景中主体尺度、运动位置和相机跟随轨迹都发生分歧。因此相对Dense的LPIPS损失包含轨迹/构图差异，不能全部解释成局部纹理损坏。帆船DASA在5秒时船的尺度增长较少，也提示需要结合运动观察一致性分数。上述仅为抽帧观察，完整30秒视频供进一步主观检查。

本次严格统一的是分辨率、视频长度、chunk大小、最大历史预算、基础骨干、文本/噪声输入和VAE。并非所有方法实际保留相同token：Forcing-KV保留原生较小上下文；我方摘要占用空间使原始历史为338页而不是360页；DASA有额外约138MiB特征缓存。原生sink、相对位置及稀疏率策略保留，没有添加自定义rebase。若要求每个方法实际保留完全相同的历史token数量，需要改变原生策略，不能将那一设置称为官方原始实现。

选择LightForcing代表分层稀疏注意力，Forcing-KV代表按head分配与缓存压缩，DASA代表区域复用与KV压缩。全部使用相同Self-Forcing基础权重进行受控推理迁移，没有复现各论文可能配套的专门微调权重；我方保持已训练的稀疏模块，骨干冻结。这是部署实现对比，不是论文榜单复现。

## 视频与前段截图

- sailboat：本地视频 `artifacts/stream_methods_20261102/sailboat/comparison.mp4`；截图 `artifacts/stream_methods_20261102/sailboat/early_contact.png`（媒体未上传）。
- skateboard：本地视频 `artifacts/stream_methods_20261102/skateboard/comparison.mp4`；截图 `artifacts/stream_methods_20261102/skateboard/early_contact.png`（媒体未上传）。
- dog：本地视频 `artifacts/stream_methods_20261102/dog/comparison.mp4`；截图 `artifacts/stream_methods_20261102/dog/early_contact.png`（媒体未上传）。

## 官方源码

- [LightForcing](https://github.com/chengtao-lv/LightForcing)：`d1e633337fff7e8a0ebf9f643dc37cd481fb299d`；兼容补丁保存于 `LightForcing.compatibility.patch`。
- [Forcing-KV](https://github.com/zju-jiyicheng/Forcing-KV)：`50ac1f454348e7f0a2e54bde71e1652b20c09fe6`；兼容补丁保存于 `Forcing-KV.compatibility.patch`。
- [DASA](https://github.com/Yuanyu0/DASA)：`4c9210d9a3eb864ac84859d0d43673a72d98ebd0`；兼容补丁保存于 `DASA.compatibility.patch`。
