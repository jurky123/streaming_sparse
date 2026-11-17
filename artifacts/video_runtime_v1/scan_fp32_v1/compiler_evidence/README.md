# ca消费边界编译证据

保存本机缓存中P432/tile72/FP32 Reader输出的生成代码。缓存中存在不同特化版本，本文件用于证明具体ca消费方式，不将两份完整wrapper逐项差异都归因于scan优化。

before.py.txt：`triton_poi_fused__to_copy_exclude_scan_27`读取CAND，写7,077,888个FP64元素buf59，随后传给exclude_scan；即28.312MB读＋56.623MB写。after.py.txt：buf49来自Reader第三个输出CAND，加入侧直接传`reinterpret_tensor(buf49, (864,16,32,16), (26112,1632,34,1), 546)`，没有物化ca。log输入仍先提升FP64。真实同轮时间证据在../scan_p432_steady_detail，不使用wrapper整体差异估算收益。
