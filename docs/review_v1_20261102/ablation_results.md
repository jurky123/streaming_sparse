# 消融实验（2026-11-03）

固定512×768、338历史页、16 probes、10%精确预算；骨干与各候选checkpoint冻结。没有为去除模块重新训练，结果表示当前模型对该模块的依赖。共享GPU不作速度结论。

## 同状态局部消融

固定Q/K/V与物理页。均值/细节/无补偿及fixed_mask训练对照使用完全相同mask；路由对照使用相同摘要和预算。局部结果不包含重排分组或闭环状态变化。

|场景|对照|事件数|平均NMSE|能量加权NMSE|相对full变差事件|
|---|---|---:|---:|---:|---:|
|horse_field|full|36|0.063058|0.094182|0/36|
|horse_field|initial_fixed_mask|36|0.063071|0.094189|34/36|
|horse_field|initial_own_route|36|0.063116|0.094252|24/36|
|horse_field|mass_route|36|0.065632|0.096115|33/36|
|horse_field|no_detail|36|0.063585|0.094741|34/36|
|horse_field|no_tail|36|0.084214|0.107132|36/36|
|horse_field|old_route|36|0.065332|0.095805|33/36|
|horse_field|pretrained_fixed_mask|36|0.063059|0.094183|18/36|
|horse_field|pretrained_own_route|36|0.063046|0.094191|21/36|
|horse_field|projected_mean|36|0.075499|0.103159|36/36|
|traffic_crossing|full|36|0.070835|0.093617|0/36|
|traffic_crossing|initial_fixed_mask|36|0.070850|0.093627|36/36|
|traffic_crossing|initial_own_route|36|0.070680|0.093549|21/36|
|traffic_crossing|mass_route|36|0.072069|0.094390|27/36|
|traffic_crossing|no_detail|36|0.071143|0.093831|35/36|
|traffic_crossing|no_tail|36|0.091797|0.107467|36/36|
|traffic_crossing|old_route|36|0.071943|0.094534|25/36|
|traffic_crossing|pretrained_fixed_mask|36|0.070835|0.093617|18/36|
|traffic_crossing|pretrained_own_route|36|0.070841|0.093621|16/36|
|traffic_crossing|projected_mean|36|0.078091|0.098909|36/36|

## 前段闭环消融

主窗口RGB[21,101)，seed101。full与旧版复用此前同输入的结果。每个消融自由演化自身轨迹；去均值/细节不改该次路由使用的完整统计，分组关闭仍保持原有partial-chunk淘汰页顺序。

|场景|对照|LPIPS↓|SSIM↑|PSNR↑ (dB)|相对full LPIPS变化|
|---|---|---:|---:|---:|---:|
|horse_field|full|0.27041|0.58193|15.480|+0.00000|
|horse_field|accepted|0.32709|0.53261|13.811|+0.05667|
|horse_field|mass_route|0.26768|0.58900|15.609|-0.00273|
|horse_field|no_detail|0.27351|0.57910|15.664|+0.00310|
|horse_field|no_downstream|0.25037|0.60877|16.097|-0.02005|
|horse_field|no_grouping|0.34227|0.48985|13.432|+0.07186|
|horse_field|no_new_training|0.26556|0.59020|15.833|-0.00485|
|horse_field|projected_mean|0.32450|0.50709|13.809|+0.05408|
|traffic_crossing|full|0.32551|0.40167|15.822|+0.00000|
|traffic_crossing|accepted|0.34068|0.33813|13.431|+0.01518|
|traffic_crossing|mass_route|0.33788|0.39517|15.634|+0.01237|
|traffic_crossing|no_detail|0.32970|0.40311|16.054|+0.00420|
|traffic_crossing|no_downstream|0.33651|0.38336|15.527|+0.01101|
|traffic_crossing|no_grouping|0.23520|0.51686|16.709|-0.09031|
|traffic_crossing|no_new_training|0.33005|0.39583|15.961|+0.00454|
|traffic_crossing|projected_mean|0.35889|0.31506|13.151|+0.03338|

projected_mean仅将完整均值替换为当前basis和旧center的rank8重建，保留相同K、细节与mask，并非完整旧kernel重现。no_new_training保留新结构但不做本轮训练；no_downstream保留摘要预训练。mass_route保持同预算且强制own页。no_grouping仅在视频中评估。

相关事件不作为独立统计样本；两个场景的消融不能支持普遍质量或速度结论。缺失组仍待运行，不填充推测结果。
