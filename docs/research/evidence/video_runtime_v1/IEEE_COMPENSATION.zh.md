> 后续用户明确接受此前近似组合为基线，已恢复BF16四乘积mode4默认；IEEE mode1保留为对照。本页记录实现与测试过程，原“只允许mode1”限制已撤销。

# 补偿改用数学等价的FP32乘法

用户要求取消补偿的BF16高低项四乘积近似。已将流式补偿的P×V改为`tl.dot(p, v, input_precision="ieee")`：输入概率和页面Value保持FP32，不压成BF16/TF32。VideoRuntime、mixed_attention和视频入口均默认tail mode1；VideoRuntime与视频入口只接受该模式。旧mode0/3/4仅保留用于隔离诊断和历史对照，不能作为新视频运行时配置。

## 数学语义

令未精算页代表的共同归一化权重为w_pr，完整页面均值为mu_p，rank8细节为d_pak，重建基为B_dk，H_ar为固定Haar系数。静态页面Value为：

V_prd = mu_pd + sum_a,k H_ar d_pak B_dk。

直接乘法输出sum_p,r w_pr V_prd，展开后等于：

sum_p (sum_r w_pr) mu_pd + sum_k [sum_p,r,a w_pr H_ar d_pak] B_dk。

这正是完整均值矩阵乘＋低秩细节补偿。没有截断或删除项，代表16 token的质量因子、精算页排除和共同归一化保持不变。新乘法取消了旧p≈p_hi+p_lo、V≈V_hi+V_lo的两项BF16表示误差。FP32乘加、Value重建、在线softmax和分片归约仍有标准浮点舍入，数学等价不意味着与原计算顺序逐位一致。

此修改只替换补偿乘法。Reader的FP32页面归约、split_summary等其他开关未变，不能宣称全部优化视频的差异已消除。原始精算分支的高低项计算也不属于这次补偿改动。

## 隔离验证

`check_ieee_compensation.py`使用固定seed701的FP32概率和Value，128组64×64乘64×128，对照相同输入的FP64乘法。旧四乘积NMSE1.12155e-11、max_abs8.14856e-5；IEEE FP32 NMSE2.08418e-14、max_abs1.03507e-5。算子0.032768→0.041984ms（新/旧耗时1.281×）。生成PTX确认IEEE乘法不含mma.sync，保留PTX与JSON证据。

真实city_walk捕获输入、新全覆盖Reader、预留432页/有效410页非连续stride的完整Attention检查通过，输出全部有限。IEEE四分片/8warps约0.555008ms，相对旧“独立精算＋完整均值/低秩补偿”局部NMSE1.44032e-7；该对照还包含归约顺序等差异，不是单独P×V乘法误差。

原始证据：`artifacts/video_runtime_v1/ieee_compensation/`。完整短视频接入验证已完成。

## 完整短视频接入

保持上一轮所有其他优化，仅补偿tail mode4→1。生成两次16.20857/16.21287秒，中位16.21072秒；旧版15.30128秒，耗时增加5.94%。latent全部有限；与旧版NMSE=0.213515，max_abs=4.42188。这是数值路径改变后闭环差异，不代表补偿公式改变，也不据此判断哪条视频质量更好。未重测VAE或15秒视频，不宣称新的端到端/图像质量结果。

真实输入旧四乘积同权重对照四分片0.412672ms，新IEEE0.555008ms，模块耗时增加34.49%。本次请求已完成；未开启下一轮优化。
