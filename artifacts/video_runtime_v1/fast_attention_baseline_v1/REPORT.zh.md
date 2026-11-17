# 当前基线：旧版快速精算与补偿

按用户明确要求，精算和补偿使用旧快速实现，作为后续优化基线。保留当前Risk、摘要权重、Full_Pset、Adaptive φ与B16；本轮不是四项旧模块全部替换。Reader使用已逐位验收A-only＋两阶段，RoPE使用已逐位验收旧融合核；PAGE32与lazy均关闭。

精算通过Page64直接LUT读取常驻KV，补偿为完整128维V均值＋rank8细节，旧版BF16两项AV，按base2 LSE合并。没有用低秩均值替代完整均值。该实现相对先前四乘积/代表向量实现有数值差异，用户已指定采用；报告如实保留对照。

## 速度

GPU3同卡顺序；单city_walk_s701，9 chunks/105帧，768×512，16FPS，360历史页。各组完整预热后两次计时，再单独阶段profile/VAE；总耗时为生成＋VAE中位数之和，不含编译/加载/视频编码。

|实现|生成/s|含VAE/s|含VAE相对旧基线|相对Dense|
|---|---:|---:|---:|---:|
|reference|12.749|17.061|1.000×|0.751×|
|current|11.325|15.626|1.092×|0.820×|
|dense|8.507|12.812|1.332×|1.000×|

## 实现切换误差（current相对reference）

{'latent_equal': False, 'latent_nmse': 0.30102160573005676, 'rgb_equal': False, 'rgb_mae_255': 13.434152649571656, 'psnr': 17.32991606558995, 'ssim': 0.6838870655745268, 'lpips': 0.23327281111851333, 'window': [21, 101]}

PSNR/SSIM/LPIPS窗口[21,101)，PSNR原分辨率，SSIM/LPIPS为256×384；RGB MAE覆盖105帧。此处没有计算算法相对Dense的质量误差。

## 阶段profile

|模块|reference/s|current/s|
|---|---:|---:|
|attention|3.212|1.851|
|route_and_correction|4.048|4.048|
|summary_and_encoding|0.491|0.447|
|clean_commit|0.894|0.878|

## 默认入口与复现

VideoRuntime的legacy_components缺省值为None，解析为attention；显式字符串仍按指定组件执行，空字符串可恢复旧控制。benchmark_hotspots参数缺省为attention；完整当前基线用--reader-layout --reader-split --legacy-components rope,attention。命令见local_setup/CURRENT_VIDEO_BASELINE.txt，成对验收脚本run_fast_attention_baseline.py。旧实验产物不修改。
