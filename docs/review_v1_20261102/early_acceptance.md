# 扩大后的闭环质量验收

相同512×768、338历史页（layer0为360）、同seed与噪声。主验收看RGB[21,101)，同时记录中后段。共享GPU，不作速度结论。

|场景|seed|窗口|旧版LPIPS|新版LPIPS|旧版SSIM|新版SSIM|
|---|---|---|---|---|---|---|
|horse_field|101|early|0.32709|0.27041|0.53261|0.58193|
|traffic_crossing|101|early|0.34068|0.32551|0.33813|0.40167|

状态：numeric_screen_passed_pending_visual_review。

FP32质量参考实现，尚未验收生产TF32路径；LPIPS/SSIM只衡量接近同输入Dense，不能替代运动、主体和闪烁的视频检查。

各case目录comparison.mp4为完整对照，左到右：Dense、旧版、新版。
