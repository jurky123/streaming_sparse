# 满窗口路由直接读取常驻摘要

432 页时，每层已有连续的 BF16 keys、stats；两者合计 5,806,080 字节。原实现每次将这些张量复制到共享路由图。新实现为未走 Full_Pset 纠正的满窗口层按常驻地址捕获图，直接读取更新后的摘要，共用固定的本页 ID，并跳过该分支恒为零的计数相加。其他窗口与纠正分支沿用旧路径。首次实验把临时 softplus 权重的地址放进图缓存键，导致重复捕获；修正后缓存键只用常驻 keys/stats 地址，权重仍在重放前写入图内的固定张量。

四个真实满窗口事件中，移除 staging 的组合模块加速 1.034–1.035×。与当前实现逐项核对 Risk、掩码、页序列，全部相同，页序列也与捕获记录一致。[模块](module.json) 独立层图的结果另在整段轨迹中验证。

city_walk/701 的路由 A/B 使用这轮之前的归一化设置；因 base 成对差值跨零，额外用两条视频在融合归一化设置下复核 base，每条 6 对。42 对的最终 latent 均逐字节一致。

| 视频 | 档位 | 对数 | 旧 / 新中位数（s） | 加速比 | 更快次数 | 成对节省中位数（ms） |
|---|---|---:|---:|---:|---:|---:|
| city_walk | [base](base.json) | 10 | 5.26293 / 5.25491 | 1.00153× | 7 | 7.07 |
| city_walk | [intersection](intersection.json) | 10 | 5.32422 / 5.31264 | 1.00218× | 10 | 7.38 |
| city_walk | [step](step.json) | 10 | 5.95943 / 5.95137 | 1.00136× | 10 | 8.46 |
| kitchen_cook | [base](repeat_kitchen_cook_703/base.json) | 6 | 5.15091 / 5.14590 | 1.00097× | 6 | 8.72 |
| tennis_match | [base](repeat_tennis_match_703/base.json) | 6 | 5.21144 / 5.20326 | 1.00157× | 5 | 8.42 |

GPU6 独立进程测首条 105 帧视频，关闭 / 启用层图分别为 9.30 / 14.78 秒。峰值 allocated 增加 47.0 MiB，满窗口层图 29 个。默认采用预热后更快的路径。只生成一条冷启动视频可用 `accepted_tiers --no-resident-route`。[冷启动与内存](cold_cost.json)

另试了共享图内加载设备地址、共享图用层索引定位整块摘要。原始地址方式明显回退，层索引方式在七个事件的页序列一致但 Risk 未逐位匹配，均保留为实验材料。[地址](addresses.json) · [地址对齐](aligned_addresses.json) · [层偏移](layer_offsets.json) · [实验代码差异](address_experiments.patch)

默认入口各预热后 5 次的中位数为 base 5.20158 秒、intersection 5.23170 秒、step 5.94019 秒，三档最终 latent 均与上一部署逐字节一致。GPU7 原生 Dense 同样启用残差融合，为 6.54073 秒；base 加速 1.25745×，距 4.67082 秒目标还差 0.53077 秒。[独立部署](deployment.json) · [时间线](device_timeline.json)
