from .run import *
import concurrent.futures,shutil,tarfile

def table(name,rows):
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def run():
 rows=[];quality=[];repeats=[];audits=[];summary=[];oracles=[];gates=[]
 for eid in EVENTS:
  rr=json.loads((OUT/eid/'result.json').read_text());rows+=rr['rows'];quality+=rr['quality'];repeats+=rr['repeat_confirmation'];audits.append(rr['audit']);truth=np.load(OUT/eid/'truth.npz');raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'));J=objective(raw,raw['risk'])
  for p in ['medium','full_legal']:
   g=truth[p+'_gain'];r=truth[p+'_remove'];a=truth[p+'_add'];assert np.max(abs(g-swap_labels(raw,raw['risk'],r,a)))<1e-10;oracles.append(dict(event_id=eid,panel=p,risk_gain_pct=0,oracle_relative_pct=100*max(0,float(g.max()))/J,oracle_remove=int(r[g.argmax()]),oracle_add=int(a[g.argmax()])))
   for b in [1,2,4]:
    for method in ['set','independent']:
     sel=[x for x in rr['rows'] if x['panel']==p and x['b']==b and x['method']==method];v=np.array([x['gated_relative_pct'] for x in sel]);actions=[(x['remove'],x['add']) if x['execute'] else (-1,-1) for x in sel];same=np.mean([actions[i]==actions[j] for i in range(5) for j in range(i+1,5)])
     summary.append(dict(event_id=eid,panel=p,b=b,method=method,mean_gain_pct=float(v.mean()),std_gain_pct=float(v.std()),minimum_gain_pct=float(v.min()),p10_gain_pct=float(np.quantile(v,.1)),positive_count=int((v>0).sum()),degraded_count=int((v<0).sum()),execution_rate=float(np.mean([x['execute'] for x in sel])),distinct_actions=len(set(actions)),pairwise_action_agreement=float(same),oracle_gain_recovery=float(np.mean([x['gated_gain']/x['oracle_gain'] for x in sel])) ))
 for b in [1,2,4]:
  passed=[]
  for eid in EVENTS:
   s=next(x for x in summary if x['event_id']==eid and x['panel']=='medium' and x['b']==b and x['method']=='set');i=next(x for x in summary if x['event_id']==eid and x['panel']=='medium' and x['b']==b and x['method']=='independent');passed.append(s['mean_gain_pct']>0 and s['positive_count']>=4 and s['mean_gain_pct']>i['mean_gain_pct'])
  gates.append(dict(b=b,information_both_pass=all(passed),low_sampling_eligible=b<=2,cost_gate_pass=False,cost_reason='Additional work measured theoretically; no demonstrated quality-matched exact-budget savings at fixedK. Do not treat lack of evidence as proof no future cheaper policy exists.'))
 costs=[]
 for b in [1,2,4]:
  tok=4*b;byt=tok*512;headread=410*byt;fl=4*16*410*tok*128
  costs.append(dict(b=b,tokens_per_page=tok,optional_BF16_cache_bytes_per_page=byt,raw_KV_read_bytes_per_headtile=headread,read_GiB_per_layer=864*headread/2**30,optional_history_cache_GiB_29layers=338*12*29*byt/2**30,current_sample_refresh_bytes_per_head=72*byt,probe_QK_AV_GFLOPs_per_layer=864*fl/1e9,read_ratio_to_41_exact_pages=headread/(41*64*512),QK_AV_ratio_to_41_exact_pages=fl/(4*64*41*64*128),medium_swap_norm_GFLOPs_lower_bound=2*128*16*512*864/1e9,full_swap_norm_GFLOPs_lower_bound=2*128*16*14760*864/1e9))
 for n,v in [('per_seed',rows),('information_quality',quality),('repeat_confirmation',repeats),('summary',summary),('oracle',oracles),('gates',gates),('costs',costs)]:table(n+'.csv',v)
 plan=np.load(OUT/'sampling_plan.npz');protocol=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'sampling_plan.npz')==protocol['plan_sha']
 for k in plan.files:assert np.array_equal(np.sort(plan[k],axis=-1),np.broadcast_to(np.arange(16),(410,4,16)))
 originals=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for n,h in ex.map(lambda n:(n,sha(DATA/n)),originals):assert originals[n]==h
 dump(OUT/'audit.json',dict(passed=True,frozen_files=len(originals),snapshot_audits=audits,sampling_nested_without_replacement=True,real_action_labels_independently_recomputed=True,policy='No Dense output, trueN/Z/residual/C used in sampling or action prediction; evaluation functions use frozen truth only.',data_files={str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*/probe_*.npz')}))
 lines=['# E22：Active Residual Probing 第一阶段','', '结论：每子组2 token（每页8 token）出现同信息集合决策的正信号：两例均5/5种子正收益，平均优于独立残差排序；每子组1 token则集合方法平均退化。全页探测成本仍高，尚未证明质量匹配下可节省更多精算工作。因此信息门槛通过、成本门槛未通过；不扩大数据或实现kernel，第二阶段未运行。','', '## 固定范围与同信息对照','', '仅两份旧快照（city_walk/horse_field、s101/c5/l15/step0/head8/t61）。Q64/Page64、410页、41精确页、own保护、原Risk初选，全部页真实KV分层均匀无放回探测；每组b=1/2/4，种子17/29/43/71/101。每组使用固定随机排列前b个形成嵌套样本，计划在运行前保存。5次随机采样不是5个独立QKV样本。','', '主候选池Medium：Risk最低的16个可移出精确页、Risk最高的32个未选页；另列全部14760动作诊断。两方法共用同一批探测分子/分母及其估计Dense输出，均只允许一次交换和no-op。Probe-Independent分数为页面平均平方残差/估计Dense分母平方，选择s_add−s_remove最大正动作；不是重排41页。Probe-Set最小化估计Dense与当前混合输出的平方差，分母随集合变化。政策不读取真实Dense输出或真实残差；真值仅用于评价。','', '## 主结果','', '相对Risk的真实Q64 NMSE改善，五种子均值；括号内为正收益种子数。','', '| 每组b / 每页token | City Independent | City Set | Horse Independent | Horse Set |','|---|---:|---:|---:|---:|']
 def s(e,b,m):return next(x for x in summary if x['event_id']==e and x['b']==b and x['method']==m and x['panel']=='medium')
 for b in [1,2,4]:
  vals=[s(e,b,m) for e in EVENTS for m in ['independent','set']];lines.append(f'| {b} / {4*b} | '+' | '.join(f"{x['mean_gain_pct']:+.3f}% ({x['positive_count']}/5)" for x in vals)+' |')
 lines+=['','同池True-Set Oracle（仅离线）收益：'+', '.join(f"{x['event_id'].split('_s101')[0]} {x['oracle_relative_pct']:.3f}%" for x in oracles if x['panel']=='medium')+'。Risk不交换为0%。','', 'b2的Set相对Independent增益：City约8.106个百分点，Horse约1.116个百分点。这里只能说明两份快照和五次随机采样下有同信息组合优势；不是跨场景统计显著性或可部署结论。b4 City独立选择仍可退化，说明更多采样并不保证动作选择单调改善。','', '## 方差、错误执行与独立复核','', '| 事件 | b | Set最差改善 | Set动作种类/5 | 动作两两一致率 |','|---|---:|---:|---:|---:|']
 for e in EVENTS:
  for b in [1,2,4]:
   x=s(e,b,'set');lines.append(f"| {e.split('_s101')[0]} | {b} | {x['minimum_gain_pct']:+.3f}% | {x['distinct_actions']} | {x['pairwise_action_agreement']:.1%} |")
 lines+=['','独立重复诊断：17提出/29复核、43提出/71复核；第二次只判断第一次同一动作的估计收益是否为正，不读取真实标签，也不另挑动作。原始执行率、退化比例、P10、排序相关性见per_seed.csv；交互符号与能量误差见information_quality.csv。全候选面板与Medium分列，不能混合Oracle口径。','', '| 事件 | b | Set双探测均值收益 | 执行/2 |','|---|---:|---:|---:|']
 for e in EVENTS:
  for b in [1,2,4]:
   x=[v for v in repeats if v['event_id']==e and v['b']==b and v['method']=='set' and v['panel']=='medium'];lines.append(f"| {e.split('_s101')[0]} | {b} | {np.mean([v['relative_pct'] for v in x]):+.3f}% | {sum(v['execute'] for v in x)}/2 |")
 lines+=['','复核只有两个预登记seed对，不能称置信度校准或安全保证。需要第二份探测，QK/AV工作翻倍；若对重叠token复用，独立两次样本期望并集大小每组为2b−b²/16，而非恰好2b；本轮没有实现复用。','', '## 成本','', '| b | 可选缓存B/页 | 探测读取/41页精算读取 | 探测QK/AV / 41页精算QK/AV | 每层探测GFLOPs |','|---|---:|---:|---:|---:|']
 for c in costs:lines.append(f"| {c['b']} | {c['optional_BF16_cache_bytes_per_page']} | {c['read_ratio_to_41_exact_pages']:.3f} | {c['QK_AV_ratio_to_41_exact_pages']:.3f} | {c['probe_QK_AV_GFLOPs_per_layer']:.3f} |")
 lines+=['','BF16 K/V每token512字节，读取按每head/tile独立扫描、不假设跨tile缓存复用；QK+AV为4×M×tokens×128 FLOPs。b2额外读取等于原41页精算KV读取的125%，QK/AV约31.25%。Medium仅交换范数下界另为1.812 GFLOPs/层，完整候选为52.235 GFLOPs/层，均不含完整I/B/分母、特征、exp、排序、访存和启动。','', '可直接读取原KV，不必重复缓存；若打包历史样本缓存以改善访问，会新增存储。当前72页样本每去噪调用更新。理论逻辑读取不等于实际HBM流量，小样本非连续访问还可能放大事务。最终41页不等于只计算41页：另有全页探测、路由评分；若先执行Risk再交换，还需新增精算与中间贡献复用，本轮没有先执行Risk两遍的设计。','', '固定预算下没有实测质量匹配的精算页减少或整体延迟收益，故成本门槛尚未通过；不能断言主动探测最终不经济，也不能据质量正信号直接进入kernel。候选页＋加权背景抽样、已选页反馈校准均标记未运行，保留下一步条件性研究。','', '## 数据与审计','', '每seed/b完整保存N/Z、ΔN/ΔZ、有符号残差、估计Dense输出、采样索引、候选评分和真实动作收益。数值统一使用每Query全摘要分母尺度；浮点计算FP64，输入原BF16。子组样本无放回、跨b嵌套；b16全token极限复现冻结真实贡献。估计集合目标与直接估计混合输出独立核对，所有选中mask真实目标、count/own通过，734个冻结文件哈希未变。详见audit.json。','', '源码local_experiments/e22_active_probing/；复现先run --setup，再对两event运行run --event --device，最后report。大数据在各event/probe_s*_b*.npz，精简包排除可复用大张量，保留采样计划、脚本、表和核验。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');pkg=OUT/'review';pkg.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.csv','.json','.md','.npz']:shutil.copy2(p,pkg/p.name)
 shutil.copytree(ROOT/'local_experiments/e22_active_probing',pkg/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(OUT/'e22_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e22_review')
 print('E22 completed',gates)
if __name__=='__main__':run()
