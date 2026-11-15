from .run import *
import concurrent.futures,shutil,tarfile

def table(n,rows):
 with (OUT/n).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def run():
 files=sorted((OUT/'events').glob('*/result.json'));assert len(files)==144;rows=[];diag=[];checks=[]
 lock=json.loads((OUT/'stageB_lock.json').read_text());assert not lock['evaluation_used']
 for p in files:
  r=json.loads(p.read_text());rows+=r['rows'];diag+=r['diagnostics'];checks.append(r['audit']);eid=p.parent.name
  assert sha(DATA/'contributions'/eid/'summary_and_route_features.npz')==r['audit']['online_sha'];assert sha(DATA/'contributions'/eid/'residuals_fp64.npz')==r['audit']['teacher_sha']
 summ=[];ds=[]
 for sp in ['train','validation','evaluation']:
  for m in ['dual_rank8','dual_full','value_rank8']:
   d=[x for x in diag if x['split']==sp and x['method']==m];ds.append(dict(split=sp,method=m,n=len(d),**{k:float(np.mean([v[k] for v in d])) for k in ['eon_nmse','eon_cosine','eon_pearson','net_nmse','net_cosine','I_sign_accuracy','B_nmse']}))
   for pol in ['set','independent']:
    x=[v for v in rows if v['split']==sp and v['method']==m and v['policy']==pol];g=np.array([v['gated_relative_pct'] for v in x]);summ.append(dict(split=sp,method=m,policy=pol,n=len(x),mean_gain_pct=float(g.mean()),median_gain_pct=float(np.median(g)),p10_gain_pct=float(np.quantile(g,.1)),worst_gain_pct=float(g.min()),improved=int((g>1e-8).sum()),degraded=int((g< -1e-8).sum()),noop=int(sum(not v['execute'] for v in x)),mean_initial_nmse=float(np.mean([v['initial_nmse'] for v in x])),mean_final_nmse=float(np.mean([v['final_nmse'] for v in x])),pooled_oracle_recovery=float(sum(v['gated_gain'] for v in x)/sum(v['oracle_gain'] for v in x)),mean_rank_spearman=float(np.mean([v['rank_spearman'] for v in x]))))
 for n,x in [('per_instance.csv',rows),('diagnostics.csv',diag),('summary.csv',summ),('diagnostic_summary.csv',ds)]:table(n,x)
 original=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for n,h in ex.map(lambda n:(n,sha(DATA/n)),original):assert original[n]==h
 dump(OUT/'audit.json',dict(passed=True,events=144,selected_masks=144*6,unchanged_frozen_files=len(original),max_direct_extrapolated_objective_error=max(v['direct_extrapolated_objective_error'] for v in checks),max_selected_gain_error=max(v['selected_true_gain_error'] for v in checks),initial_and_summary_unchanged=True,teacher_fields_used_only_for_metrics=True,stageB_lock_sha=sha(OUT/'stageB_lock.json')))
 costs=dict(extra_permanent_cache_bytes=0,extra_original_KV_read_bytes=0,optional_extra_summary_V_bytes_per_page=304,optional_V_read_GiB_per_layer_naive=304*410*864/2**30,reused_R4_QK_GFLOPs_per_layer=2*16*410*4*128*864/1e9,rank8_response_two_views_GFLOPs_lower_bound=2*16*410*6*8*864/1e9,rank8_candidate_norm_GFLOPs_lower_bound=2*16*512*8*864/1e9,full128_candidate_norm_GFLOPs_lower_bound=2*16*512*128*864/1e9,note='Arithmetic components only; exclude projection of fullVmean, centering, mass/denominator ops, exp forR2, selection, scans, workspace, kernel/traffic overhead. R4QK reused not extra; no speed measured. FullV prototype not fused rank8 implementation.')
 dump(OUT/'costs.json',costs)
 lines=['# E23-A：冻结四代表的双视图信息诊断','', '阶段A完成144例；阶段B未运行。开发集主配置dual_rank8 Set平均退化0.561%，退化率44.44%，未通过预登记门槛。结果否定本轮无训练双视图读出，不证明联合训练的四代表必然无效，也不是现有摘要所有可能读出的信息上限。','', '## 固定设计','', '四代表仍执行原近似Attention和原Risk，不修改物理页、摘要或预算。R2将固定相邻代表(0,1)/(2,3)的Key和Value做算术平均，各代表质量32；其logit由已有R4 logits取均值，无新QK扫描。不得用两个R4指数权重相加冒充新的R2近似。','', '修正取ΔN=N4−N2、ΔZ=Z4−Z2，等价将Raw贡献外推为2R4−R2，系数与符号固定1，没有事后调参。先形成在线中心残差e_on，再由所有页面外推贡献计算估计Dense中心，修正为一致的净残差进行集合目标评价；直接把在线中心残差当精确Dense残差会遗漏中心变化，本轮已显式处理。','', '主配置dual_rank8为现有per-head basis的正交子空间投影；dual_full为完整128维诊断；value_rank8为已有中心化摘要Value贡献、分母修正置0的基线。每组同信息比较Set与Independent（页面平方净残差排序），同一Medium候选池、一次交换、正预测收益才执行，否则no-op。M16评分，真实Q64评价。未搜索pairing、rank、系数、阈值或网络。','', '划分：36city101训练描述、36city303开发决定、72horse冻结评价。先完成开发集并锁定B不启动，再运行horse。无参数拟合，完整V诊断不能反向覆盖主配置门槛。','', '## 真实修正收益','', '改善均相对未修改的旧Risk。初选与旧Risk逐实例相同，所以修正收益与最终相对旧基线收益一致。','', '| 表示/决策 | 开发均值 | 评价均值 | 评价改善/退化/no-op | 评价P10 |','|---|---:|---:|---|---:|']
 for m in ['dual_rank8','dual_full','value_rank8']:
  for pol in ['set','independent']:
   v=next(x for x in summ if x['split']=='validation' and x['method']==m and x['policy']==pol);e=next(x for x in summ if x['split']=='evaluation' and x['method']==m and x['policy']==pol);lines.append(f"| {m}/{pol} | {v['mean_gain_pct']:+.3f}% | {e['mean_gain_pct']:+.3f}% | {e['improved']}/{e['degraded']}/{e['noop']} | {e['p10_gain_pct']:+.3f}% |")
 lines+=['','无真实标签门控，无按评价集调阈值。详细NMSE、最差退化、Oracle恢复、收益排序与动作身份见summary.csv和per_instance.csv。完整128维若局部更好只能说明投影或评分存在差异，不能自动认为满足低成本双用途表示要求。','', '## 有符号信息诊断（评价72例均值）','', '| 表示 | 在线残差NMSE | 在线残差余弦 | 在线残差Pearson | 交互符号准确率 | B NMSE |','|---|---:|---:|---:|---:|---:|']
 for d in ds:
  if d['split']=='evaluation':lines.append(f"| {d['method']} | {d['eon_nmse']:.3f} | {d['eon_cosine']:.3f} | {d['eon_pearson']:.3f} | {d['I_sign_accuracy']:.2%} | {d['B_nmse']:.3f} |")
 lines+=['','rank8残差指标在相同投影空间中比较，full128在完整空间；不得把投影后的较低误差当完整输出保真。I/B对照均为匹配16 probes的真实量，最终动作收益仍为全部64 Query。R4/R2差可观测并不等于Raw64/R4误差可预测；尤其遗漏的共同信息在双视图中可能不存在。','', '## 成本边界','', f"不增加永久摘要存储或原KV读取；可选读取已有mean/detail304字节/页，稳态每层按无跨tile复用估算{costs['optional_V_read_GiB_per_layer_naive']:.3f}GiB。已有R4 QK约{costs['reused_R4_QK_GFLOPs_per_layer']:.3f}GFLOPs/层属于复用项，不是新开销。两视图rank8响应主要乘加下界{costs['rank8_response_two_views_GFLOPs_lower_bound']:.3f}GFLOPs/层，512动作的rank8范数下界{costs['rank8_candidate_norm_GFLOPs_lower_bound']:.3f}GFLOPs/层。",'', '这些是部分算术账，未含完整V均值投影、中心化、质量/分母计算、R2指数、选择与中间状态/访存。原型为FP64诊断，不是fused低维实现；没有报告实际加速比，也不声称总新增计算一定小于几GFLOPs。','', '## 阶段B与下一步','', 'B明确未运行：主读出的开发收益与错误执行率未达标，本轮未获得进入联合训练的可靠决策信号。没有用冻结旧残差监督变化后的摘要。两份raw快照也不足以继承144例的训练/开发/评价划分；若另行开展联合训练，必须对每次新摘要重新计算Nhat/Zhat与标签，并分别验证新Risk初选、修正收益和相对旧Risk的最终收益。','', '该结论不否定双用途代表的研究方向：当前只试固定相邻合并、系数1和明确的集合读出。不能将本轮负结果升级为“四代表没有信息”或“联合训练无效”。','', '## 核验和复现','', f'144例、864个选中mask独立真实FP64目标及count/own检查；外推分子分母直接混合目标与残差评分一致。{len(original)}个冻结源文件哈希不变。完整审计见audit.json。源码local_experiments/e23_dual_representatives/；运行setup、development各worker、写stageB_lock、evaluation各worker、report。输出逐实例scores.npz保留真值和候选集合；精简包含报告、表、协议与代码。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');pkg=OUT/'review';pkg.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.md','.csv','.json']:shutil.copy2(p,pkg/p.name)
 shutil.copytree(ROOT/'local_experiments/e23_dual_representatives',pkg/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(OUT/'e23_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e23_review')
 print('complete',[(x['method'],x['policy'],x['mean_gain_pct']) for x in summ if x['split']=='evaluation'])
if __name__=='__main__':run()
