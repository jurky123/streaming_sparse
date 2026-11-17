# 9d4d33d 基线批量短视频质量评估

14个已有场景×seed701/702×Dense/C+Risk/C+Full_Pset，共84个9-chunk视频；28组严格配对。768×512、105帧、16FPS、history360、tile72。GPU0–4每卡一个常驻进程；质量清单冻结9d4d33d代码，仅修正batch参数透传和README。P2/scan候选没有进入本清单。

源代码快照见 frozen_source.json，执行身份见 execution_identity.json。checkpoint内容哈希、同prompt/seed/长度/生成配置一致；全覆盖C_s17与Full_Pset_s17。14个场景来自现有embedding库，不称为独立未见场景测试。21–100帧的80帧指标用于主比较；前段为[21,61)，后段为[61,101)，不是长视频后段。

## 配对结果

|方法|整体LPIPS↓|前段LPIPS↓|后段LPIPS↓|整体SSIM↑|整体PSNR/dB↑|
|---|---:|---:|---:|---:|---:|
|Dense|—|—|—|—|—|
|C摘要＋Risk|0.3181|0.2130|0.4232|0.4768|14.285|
|C摘要＋Risk＋Full_Pset|0.3173|0.2061|0.4284|0.4828|14.369|

Full_Pset在19/28个配对视频上降低LPIPS。整体平均变化-0.000838（相对C+Risk约-0.26%），均值收益很小、场景差异明显，不能把此前局部Attention收益直接视为视频质量同比例收益。SSIM平均变化+0.006058，PSNR平均变化+0.084dB。

|seed|C+Risk LPIPS|Full_Pset LPIPS|
|---|---:|---:|
|701|0.3089|0.3006|
|702|0.3273|0.3339|

## 变化最大的配对视频

下表delta为Full_Pset−C+Risk，LPIPS负值表示更接近Dense。对应三列视频从左到右Dense、C+Risk、Full_Pset。

|场景|seed|LPIPS delta|SSIM delta|PSNR delta/dB|并排视频|
|---|---:|---:|---:|---:|---|
|kitchen_cook|702|-0.0320|+0.0911|+0.956|[查看](videos/kitchen_cook_s702/chunks9/comparison.mp4)|
|snow_ski|701|-0.0237|+0.0270|+0.765|[查看](videos/snow_ski_s701/chunks9/comparison.mp4)|
|tennis_match|701|-0.0184|+0.0043|+0.012|[查看](videos/tennis_match_s701/chunks9/comparison.mp4)|
|bird_flight|702|+0.0348|-0.0115|-0.658|[查看](videos/bird_flight_s702/chunks9/comparison.mp4)|
|tennis_match|702|+0.0400|-0.0565|-0.492|[查看](videos/tennis_match_s702/chunks9/comparison.mp4)|
|reef_fish|702|+0.0565|-0.0286|-1.127|[查看](videos/reef_fish_s702/chunks9/comparison.mp4)|

## 测量和产物说明

LPIPS为AlexNet、SSIM在256×384双线性缩放后计算；PSNR使用原分辨率逐帧MSE。先算每个视频指标，再对28视频等权平均；PSNR不是所有帧MSE合并后再取log。前后段直接从保存的逐帧指标切分，不重新解码。Dense在每个prompt/seed上只生成和解码一次，其像素缓存供两种稀疏方法复用。

所有84生成和84后处理任务完成，无nonfinite评分。保存clean latent、最终视频、逐帧指标和每方法对Dense的双列视频；comparison.mp4从已有编码视频拼接，无额外VAE解码。耗时含冷编译、IO和不同任务顺序，且多GPU并行运行，因此本清单不报告单视频加速比。

这批结果主要回答短视频闭环差异；主体/运动/闪烁的因果结论不能单凭LPIPS推出。先检查改善和变差最大的并排视频，再决定后续质量测试范围。没有启动15s或30s视频。

启动时冻结目录的wan_models符号链接导致local-assets相对路径解析错误；在生成任何视频前修正为真实目录＋模型子目录链接后重试，未更改算法或权重。最终展开的batch命令均显式tile72/history360，既有identity记录实际配置。
