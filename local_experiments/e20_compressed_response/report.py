from .core import *
import concurrent.futures,shutil,tarfile

def run():
 rows={k:[] for k in ['performance','page_metrics','selected']};audits=[];phase=[]
 for eid in EVENTS:
  a=json.loads((OUT/'stage1'/eid/'result.json').read_text());phase.extend(x for m in a['methods'] for x in m['performance']);r=json.loads((OUT/'compression'/eid/'result.json').read_text());audits.append(r['audit'])
  for m in r['methods']:
   for k in rows:rows[k].extend(m[k])
 for k,v in rows.items():table(k+'.csv',v)
 table('stage1.csv',phase)
 def get(e,m,scope='all_pages',M=16):return next(x for x in rows['performance'] if x['event_id']==e and x['method']==m and x['scope']==scope and x['probes']==M and x['panel']=='full_legal')
 costs=[];screens=[];changes=[]
 for name,k,v in CONFIGS:
  rv=8 if v.endswith('8') else 16;s=k*(k+1)//2;b=8*(128+(1+rv)*(k+s));flops=8*(1+rv)*(k+s)+256*rv
  costs.append(dict(config=name,bytes_per_page=b,history_bytes_per_head=338*b,history_MiB_per_layer=338*b*12/2**20,history_GiB_29_layers=338*b*12*29/2**30,naive_read_GiB_per_layer_call=338*b*72*12/2**30,shared_basis_bytes_per_head=128*(k+rv)*4,moment_contraction_flops_per_query_page=flops,M16_layer_GFLOPs=(338*16*flops+16*256*k)*864/1e9,Q64_layer_GFLOPs=(338*64*flops+64*256*k)*864/1e9,commit_dominant_flops_per_page=2*64*128*(k+rv)+2*64*(1+rv)*(k+s),current_refresh_bytes_per_head=72*b))
  for scope in ['all_pages','history_only']:
   rr=[get(e,name+'_BF16',scope)['gated_relative_pct']/get(e,'full_M2')['gated_relative_pct'] for e in EVENTS];screens.append(dict(config=name,scope=scope,city_M2_retention=rr[0],horse_M2_retention=rr[1],passes_both_80pct=all(t>=.8 for t in rr)))
 for x in rows['performance']:
  if x['panel']!='full_legal':continue
  b=get(x['event_id'],'full_M2',M=x['probes']);changes.append(dict(event_id=x['event_id'],method=x['method'],scope=x['scope'],probes=x['probes'],M2_gain_pct=b['gated_relative_pct'],gain_pct=x['gated_relative_pct'],M2_positive_compressed_negative=b['gated_relative_pct']>0 and x['gated_relative_pct']<0,retention=x['gated_relative_pct']/b['gated_relative_pct']))
 table('costs.csv',costs);table('screen.csv',screens);table('decision_changes.csv',changes)
 original=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for n,h in ex.map(lambda n:(n,sha(DATA/n)),original):assert h==original[n]
 n=0
 for e in json.loads((E13/'preparation.json').read_text())['events']:
  for name,h in e['files'].items():assert sha(E13/'events'/e['event_id']/name)==h;n+=1
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['checkpoint_sha']==sha(CHECKPOINT) and protocol['E19_PCA_sha']==sha(E19/'pca16.npz')
 cache_hashes={}
 for p in (OUT/'compression').glob('*/*_BF16.pt'):
  ss=torch.load(p,weights_only=True);assert all(t.dtype==torch.bfloat16 for t in ss.values());cache_hashes[str(p.relative_to(OUT))]=sha(p)
 dump(OUT/'audit.json',dict(passed=True,frozen_files=len(original),E13_frozen_files=n,checks=audits,cache_hashes=cache_hashes,all_performance_valid=all(x['status']=='VALID' for x in rows['performance'])))
 lines=['# E20：低秩二阶响应压缩诊断','', '结论：本轮四个 BF16 配置均未通过“两例都保留至少80%完整 M2 收益”的预登记筛选。Value 单独压缩较好，主要观察到的损失来自这套固定 Key 基；这不能推断所有低秩 Key 表示都不可行。仅历史页增加统计又进一步损失收益，暂不进入正式缓存或 kernel 实现。','', '## 范围与选择规则','', '仅两个原始快照：city_walk/horse_field，seed101、chunk5、layer15、step0、head8、tile61。Q64/Page64、410页、41精确页、own保护、原Risk初始集合；完整14760个合法交换，另外保留共同面板。所有真实收益均用原始64 Query目标独立复核。没有新视频、训练或生产推理修改。','', '中心化 M2 在city与原M2同为11.2656%收益，但Risk未选页残差NMSE为0.45183，原M2为0.43824，因此依city独立规则锁定原M2。horse中心化收益7.3255%（原4.6041%）只报告，不反向更改形式。','', 'Key基来自冻结layer15/head8 Key-response bank的QR前4/8列，不在快照拟合。Value使用现有basis的QR或E19训练集冻结的V-response PCA前8/16列。全128维a0保留；对称矩只存上三角。主结果使用实际BF16统计、FP32共享基，在FP64进行诊断收缩，不代表部署kernel精度或速度。','', '## 真实单次交换收益','', '下表为相对初始Risk的NMSE降低百分比；负数为退化，M16评分、Q64真实评价。','', '| 配置 | 字节/页 | City 全页 | Horse 全页 | City 仅历史 | Horse 仅历史 |','|---|---:|---:|---:|---:|---:|']
 for name,b in [('full_M2',None)]+[(n,next(c['bytes_per_page'] for c in costs if c['config']==n)) for n,_,_ in CONFIGS]:
  m=name if b is None else name+'_BF16';vals=[get(e,m,sc)['gated_relative_pct'] for sc in ['all_pages','history_only'] for e in EVENTS];lines.append('| '+name+' | '+str(b or '离线完整矩')+' | '+' | '.join(f'{x:+.4f}%' for x in vals)+' |')
 lines+=['','仅历史版本保留原Risk、全部候选和真实当前KV评价；当前页新增残差修正置零、估计精算质量等于原摘要质量。它是“当前不增加统计”的具体诊断，不代表所有当前页误差估计方法。选中动作的当前/历史身份见selected.csv。','', '## 分离压缩损失','', '| 诊断 | City M16 | Horse M16 | City Q64 | Horse Q64 |','|---|---:|---:|---:|---:|']
 for m in ['K_only4','K_only8','V_only_current8','V_only_pca8','V_only_pca16']:
  vals=[get(e,m,M=M)['gated_relative_pct'] for M in [16,64] for e in EVENTS];lines.append('| '+m+' | '+' | '.join(f'{v:+.4f}%' for v in vals)+' |')
 lines+=['','BF16与FP64控制的所有选中动作一致；统计量量化未改变本轮决策。K4/PCA16使horse从M2正收益转为负收益；K8组合在city只保留约46%–50%的M2收益。Value单独压缩在两例保持正收益，但不能据此保证跨状态稳定。Q64与M16完整结果、交互符号准确率、I/B误差、排序Spearman和regret见performance.csv；未选页净残差、ΔN/ΔZ误差见page_metrics.csv。','', '## Streaming成本','', '| 配置 | 历史缓存/层 MiB | 29层 GiB | 无跨tile复用读取/层调用 GiB | M16矩收缩 GFLOPs/层 |','|---|---:|---:|---:|---:|']
 for c in costs:lines.append(f"| {c['config']} | {c['history_MiB_per_layer']:.2f} | {c['history_GiB_29_layers']:.3f} | {c['naive_read_GiB_per_layer_call']:.3f} | {c['M16_layer_GFLOPs']:.3f} |")
 lines+=['','缓存为338历史页×12heads，不含原始KV、现有摘要、共享基、对齐和分配开销；不是替换原始KV的节省。读取账按72tiles×12heads扫描，真实缓存复用可能降低HBM读取，未实测。每Query每页主要矩收缩为8(1+rV)(rK+rK(rK+1)/2)+256rV FLOPs，另计共享Query投影；不含128维仿射项、exp、动作排序、访存和kernel启动。当前72页若新增统计需每步刷新，历史可commit复用。','', '本轮完整交换枚举仍是离线搜索，不是低成本在线路由。仅候选残差范数就至少约2×128×M×14760 FLOPs/head-tile，M16约0.0605 GFLOPs/head-tile、52.2 GFLOPs/层，尚不含其他动作运算。不能以小矩收缩成本代替完整路由成本。','', '## 核验和交付','', f'原始M2评分与E19一致；联合FP64统计收缩与直接展开最大差≤1.12e-16；152个压缩阶段选中mask独立FP64目标复核，count/own通过。{len(original)}个冻结数据文件及{n}个E13文件哈希未变。所有混合分母有效，无裁剪或删除异常候选。详见audit.json。','', '后续建议优先研究更合适的共享Key子空间；本轮不支持把这套低秩统计直接接入正式Streaming推理。只有两个同层同head快照，任何正负结果都不作总体泛化结论。','', '复现：先运行core.setup()，stage1分别指定两个event/device，然后select_form，compress分别运行两例，最后report。已有form_lock不可在观察horse后重选。全部脚本在local_experiments/e20_compressed_response/；完整统计缓存和评分保留在artifacts/e20_compressed_response_v1/compression/。精简包不包含原始QKV或可再生大张量。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 pkg=OUT/'review';pkg.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.md','.csv','.json','.npz']:shutil.copy2(p,pkg/p.name)
 shutil.copytree(ROOT/'local_experiments/e20_compressed_response',pkg/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(OUT/'e20_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e20_review')
 print('complete',OUT/'REPORT.md')
if __name__=='__main__':run()
