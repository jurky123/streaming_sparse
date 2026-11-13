# 扩大后的闭环质量验收

相同512×768、338历史页（layer0为360）、同seed与噪声。主验收看RGB[21,101)，同时记录中后段。共享GPU，不作速度结论。

|场景|seed|窗口|旧版LPIPS|新版LPIPS|旧版SSIM|新版SSIM|
|---|---|---|---|---|---|---|
|horse_field|101|early|0.32709|0.27041|0.53261|0.58193|
|horse_field|101|middle|0.53551|0.65498|0.28395|0.10291|
|horse_field|101|late|0.51239|0.56945|0.10990|0.09606|
|horse_field|303|early|0.33542|0.29023|0.40011|0.47102|
|horse_field|303|middle|0.54324|0.47921|0.17550|0.20768|
|horse_field|303|late|0.56668|0.53569|0.09992|0.11806|
|traffic_crossing|101|early|0.34068|0.32551|0.33813|0.40167|
|traffic_crossing|101|middle|0.56640|0.58373|0.09219|0.13144|
|traffic_crossing|101|late|0.71549|0.63281|0.04935|0.11525|
|traffic_crossing|303|early|0.38395|0.35839|0.34975|0.39099|
|traffic_crossing|303|middle|0.55016|0.54111|0.08808|0.10975|
|traffic_crossing|303|late|0.58866|0.60162|0.06961|0.09610|

状态：numeric_screen_passed_pending_visual_review。

FP32质量参考实现，尚未验收生产TF32路径；LPIPS/SSIM只衡量接近同输入Dense，不能替代运动、主体和闪烁的视频检查。

各case目录comparison.mp4为完整对照，左到右：Dense、旧版、新版。
