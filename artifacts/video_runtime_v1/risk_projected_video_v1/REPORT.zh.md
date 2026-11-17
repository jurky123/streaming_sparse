# 保留原投影的Risk融合验收

保留原FP32投影SGEMM，只融合其后的Risk特征及M16归约；完整权重、算法配置、候选规则、精算/补偿均不变。与此前旧Risk融合不同，不把8维投影改成Triton顺序累加。最终BP16/WARPS8、feature顺序相加、libdevice sqrt，未降精度。

24组布局/算术次序/拆分M16对照后选择融合版本，再做4种窗口复验。局部Risk不是逐位一致，NMSE约3–5e-15；初选mask、移出池和加入池均一致。M16拆分没有改善误差且变慢，不采用。

## 隔离Risk

|页数|前/ms|后/ms|加速|NMSE|
|---:|---:|---:|---:|---:|
|216|0.500|0.237|2.113×|4.34e-15|
|288|0.675|0.310|2.175×|4.37e-15|
|360|0.827|0.370|2.238×|4.87e-15|
|432|1.172|0.456|2.573×|3.27e-15|

风险数值量级较大，432页最大绝对差0.125不能脱离NMSE单独解读；所有固定输入路由池契约已核对。

## 完整路由与视频

完整路由4.996→4.911ms（1.017×），远小于隔离Risk倍数。原编译器把Risk与Reader log质量计算融合在一个约0.660ms的kernel中，并有特征stack/mean等后处理；此前“Risk约0.875ms”的表格标签包含共享工作且不是全部Risk开销，不能当作纯Risk独占时间。本轮拆分后Reader FP64 logmass约0.560ms重新显露，Risk三项主要kernel为0.172ms归约＋0.230ms概率/查询预处理＋0.066ms投影。

GPU3同卡顺序，city_walk_s701、9chunks/105帧、768×512、16FPS、360历史页。每组完整预热后两轮计时，VAE单独预热；合计不含编译、加载、编码。两组都开旧快Attention、A-only/split Reader和scan-exclude，PAGE32/lazy关闭。

|实现|生成/s|含VAE/s|
|---|---:|---:|
|baseline|11.027|15.336|
|projected|10.948|15.261|

生成加速1.0072×，含VAE加速1.0049×。latent/RGB逐位一致，RGB差0、LPIPS0、PSNR100dB（零差封顶），SSIM实现输出0.999999572。基线latent与上一轮scan输出逐位一致。

|阶段|baseline/s|projected/s|
|---|---:|---:|
|attention|1.851|1.851|
|route_and_correction|3.757|3.676|
|summary_and_encoding|0.444|0.446|
|clean_commit|0.876|0.877|

## 下一热点和记录

当前Reader log质量kernel三次遍历页面，生成代码含13处exp和4处log（静态表达式位置，不是硬件指令计数），重复形成R4局部logsumexp。编译证据在compiler_evidence/；下一步检查是否值得缓存局部log质量再归一化。不能仅看算术量就预测收益，要比较完整路由中的融合得失。

risk_projected.py为保留投影的融合实现；runtime.py/benchmark_hotspots.py新增可选projected_risk；当前命令显式加入--projected-risk，通用参考默认关闭。短视频运行脚本run_risk_projected_video.py，隔离脚本check_risk_projected.py。暂停/恢复后检查进程均已退出且结果齐全，没有重启已完成任务。
