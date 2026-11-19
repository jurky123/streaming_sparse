# 第七轮：端点矩评分、摘要复用与Reader中间向量

用户重新激活1.2×原生Dense端到端目标后继续。仅GPU6–7，单视频city_walk/seed701，短视频105帧/9chunks与15秒240帧/21chunks，768×512、16FPS、360历史页＋72当前页。固定算法与权重不变。

## 已完成优化

端点矩评分在FP64计算E·u、范数和端点交叉内积；五组正质量归一化保持，取消误差/主导质量/微小增益等危险动作继续回退原向量评分。使用新全覆盖Reader SHA656a9d的隔离补测0.641024→0.565248ms，1.134×，最大分数差1.73e-17，动作/no-op一致；旧Reader的14个E49稳定性用例此前已通过。

复用已有split_summary算子，原生KV stride当前摘要1.504256→0.387072ms（3.886×），clean摘要1.849344→0.460800ms（4.013×）。完整V均值逐位一致；其余摘要字段局部最大NMSE约2.80e-10。仍保留原GraphSummary实现。

组合版短视频生成17.35297→15.46193秒。profile显示路由6.433→5.735秒、摘要编码1.377→0.495秒、clean commit1.234→0.897秒。优化前后latent NMSE0.205945、最大绝对差4.125，全部有限；**闭环误差验收未定**。不能以局部误差小宣称视频等价。

新Reader profile（6tiles、410页、同一个捕获事件）页面归约320us，占评分模块GPU时间57%。`reader_late.py`把页面16维响应和近似输出推迟到标量权重完成后再构造，缩短中间向量同时存活的时间，不改变算术公式。隔离0.710656→0.673792ms（1.055×），评分逐位一致；该进程绝对耗时与上一进程不同，只使用同进程A/B比值。

Reader延后构造的完整短视频生成15.30128秒（两次15.30138/15.30118），对上一组合版latent逐位一致。VAE4.27538秒，阶段中位数和19.57666秒，相对原生Dense13.13962秒为0.6712×，目标尚未达到。相比保守route配置40.69932＋4.30478=45.00410秒，生成＋VAE约2.299×。这属于同算法实现加速，不代表误差已验收。

最新短视频profile：Attention3.211秒、route5.599秒、摘要编码0.493秒、clean commit0.894秒。保持moments、split_summary、late_reader显式开关；失败的整块Graph不启用。

## 全部优化与保守优化对照（已完成）

保守组采用已有formal_short_15/short/original与formal_15_30/15s/original数据，mode=route、tile_batch=6：常驻KV、clean编码缓存、块稀疏精算、GPU路由编译保留；原补偿和FP64评分保留；不启用TF32分解GEMM、FP32Reader页面归约、流式混合Attention、split_summary或扩大tile_batch。

全部优化组为本轮mode=stream、tile_batch=72、BF16精确输入QK、四乘积tail/8warps、FP32Reader归约、packed/moment评分、split_summary、late_reader。双方C/Reader/phi/B16/base SHA、视频配置由对比脚本强制检查一致。新指标仅比较这两组，不比较算法与Dense的质量。

`compare_optimization.py`对未压缩uint8 RGB在原始768×512计算PSNR、SSIM、AlexNet LPIPS，RGB报告0–255单位MAE、RMSE、最大绝对差和R/G/B分量。SSIM用11×11高斯sigma1.5、总体协方差、valid边界；PSNR由窗口总体MSE计算；LPIPS逐帧平均。全段及去除初始21帧均统计；15秒额外统计0:80/80:160/160:240。保存逐帧指标和左保守/右全部优化并排视频。

指标自检发现默认TF32卷积导致相同帧SSIM约0.95，初次结果无效，已移到metrics_tf32_invalid.json。指标脚本关闭matmul与cuDNN TF32，并断言相同帧SSIM接近1；正式结果采用修正后重测。

完整两种时长对照、性能和逐段指标见 [全部优化与保守优化的视频差异](OPTIMIZATION_VIDEO_COMPARISON.zh.md)。15秒生成40.56171秒，VAE10.20374秒，总50.76545秒，相对Dense0.6428×。Reader延后构造的15秒latent也逐位一致。用户要求完成本轮后停止；本轮已完成，下一轮未启动。
