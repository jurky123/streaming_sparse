from .run import *
import concurrent.futures,shutil,tarfile

def csvout(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def run():
 perf=[];pages=[];selected=[];responses=[];life=[];err=0
 for eid in EVENTS:
  raw=dict(np.load(DATA/'contributions'/eid/'residuals_fp64.npz'))
  for tag in ['rank8','rank16']:
   rr=json.loads((OUT/eid/(tag+'.json')).read_text());assert rr['bases_sha']==sha(OUT/'bases.npz');responses+=rr['response'];err=max(err,rr['selected_audit_max'])
   for m in rr['methods']:perf+=m['performance'];pages+=m['page_metrics'];selected+=m['selected']
  ll=json.loads((OUT/eid/'lifecycle.json').read_text());life+=ll
  for row in ll:
   x=raw['risk'].copy();x[row['oracle_remove']]=False;x[row['oracle_add']]=True;g=objective(raw,raw['risk'])-objective(raw,x);assert abs(max(0,g)-row['oracle_gain'])<1e-10
  # Independently check ALL real candidate labels using direct residual reconstruction.
  from ..e12_swap_prediction.core import swap_labels
  r=np.flatnonzero(raw['risk']);r=r[r!=int(raw['own'])];a=np.flatnonzero(~raw['risk']);r,a=np.repeat(r,len(a)),np.tile(a,len(r));g=swap_labels(raw,raw['risk'],r,a);hist=np.load(DATA/'contributions'/eid/'summary_and_route_features.npz')['page_is_history'];groups=np.array([('H' if hist[x] else 'C')+('H' if hist[y] else 'C') for x,y in zip(r,a)])
  for row in ll:
   ix=np.ones(len(r),bool) if row['group']=='ALL' else groups==row['group'];assert int((g[ix]>0).sum())==row['positive_true_actions'];assert abs(max(0,g[ix].max())-row['oracle_gain'])<1e-10
 for name,rows in [('performance',perf),('page_metrics',pages),('selected',selected),('query_key_response',responses),('lifecycle',life)]:csvout(name+'.csv',rows)
 def get(e,m,M=16):return next(x for x in perf if x['event_id']==e and x['method']==m and x['panel']=='full_legal' and x['probes']==M)
 costs=[]
 for k in [8,16]:
  for v in [8,128]:
   b=8*(128+(1+v)*(k+k*(k+1)//2));fl=8*(1+v)*(k+k*(k+1)//2)+256*v;costs.append(dict(key_rank=k,value_rank=v,bytes_per_page=b,history_GiB_29layers=338*12*29*b/2**30,read_GiB_layer_call=338*864*b/2**30,M16_contraction_GFLOPs=(338*16*fl+16*256*k)*864/1e9,quality_tested_fullV_only=v==128))
 csvout('hypothetical_costs.csv',costs)
 lock=json.loads((OUT/'basis_lock.json').read_text());assert lock['bases_sha']==sha(OUT/'bases.npz');assert lock['checkpoint_sha']==sha(CHECKPOINT);cov=np.zeros((128,128))
 for eid,h in lock['train_sources'].items():
  p=DATA/'contributions'/eid/'summary_and_route_features.npz';assert sha(p)==h;assert eid.startswith('city_walk_s101_');q=np.load(p)['query'].astype(float);cov+=q.T@q/(64*36)
 bb=np.load(OUT/'bases.npz');ev=bb['trainQ_spectrum'];U=bb['trainQ16'];res=float(np.max(abs(cov@U-U*ev[:16])));assert res<1e-5
 orig=json.loads((DATA/'delivery_checksums.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for n,h in ex.map(lambda n:(n,sha(DATA/n)),orig):assert h==orig[n]
 dump(OUT/'audit.json',dict(passed=True,frozen_files=len(orig),train_query_events=36,train_basis_eigen_residual=res,max_A_selected_gain_error=err,full_B_action_labels_verified=2*14760,bankQR8_matches_E20=True,noop_count_own_checked=True,source_scope='Oracle raw K fitting only per snapshot and labeled Oracle; no heldout query fit'))
 lines=['# E21：Key子空间与历史/当前KV诊断','', '结论：本轮没有找到通过双实例质量门槛的训练阶段固定Key基。唯一rank16对照在city恢复完整M2收益，在horse只剩0.489%。暂时停止二阶矩压缩路线的缓存、kernel和配置扩展。这个结论只针对本轮有限基，不是所有低秩响应表示的不可能性证明。','', '## 范围与预登记','', '仅复用E20两份seed101/chunk5/layer15/step0/head8/tile61快照，Q64/Page64/P410/K41/own保护，原Risk，14760个全合法动作及原共同面板。固定原M2、完整128维Value响应，FP64矩诊断、FP32共享基，无Value压缩、无新模型或新视频。M16评分为主、Q64评分为信息对照，所有真实收益都以完整64 Query误差计算。','', 'bankQR8复用E20；bankSVD8对完整bank32按奇异值排序。trainQ8/16使用36个city_walk seed101训练事件的全部64 Query等事件权重、未中心化二阶矩，不读取city303/horse拟合；它是跨head共享基，不是专门为head8拟合。City快照属于训练分布，因此不能视为独立泛化测试。snapshotKOracle8按各快照真实k−κ二阶矩拟合，只是数据自适应容量探针，并非最优Query-response子空间或可泛化方法。','', '预先规定：非Oracle rank8均未通过两例≥80%FullM2收益时，只增加同一训练Query谱的rank16；触发记录见rank16_trigger.json。未追加其他rank或依据horse调整基。','', '## A：真实交换收益与Key响应误差','', '收益为相对Risk的NMSE降低百分比。δ NMSE比较投影前后的qᵀ(k−κ)/√128，不是Key重建误差。','', '| Key基 | City δ NMSE | Horse δ NMSE | City M16收益 | Horse M16收益 | City Q64收益 | Horse Q64收益 |','|---|---:|---:|---:|---:|---:|---:|']
 for m in ['bankQR8','bankSVD8','trainQ8','snapshotKOracle8','trainQ16']:
  ds=[next(x['delta_nmse'] for x in responses if x['event_id']==e and x['method']==m) for e in EVENTS];gs=[get(e,m,M)['gated_relative_pct'] for M in [16,64] for e in EVENTS];lines.append('| '+m+' | '+' | '.join([f'{x:.4f}' for x in ds]+[f'{x:+.4f}%' for x in gs])+' |')
 lines+=['','完整M2 M16参考：City +11.2656%，Horse +4.6041%。bankSVD8改善City δ误差却没有改善动作收益；trainQ16在horse仍有较大响应误差。snapshotKOracle8是Key方差PCA，不保证保留Query加权响应，因此其失败不能作为rank8理论容量上限。Horse该Oracle从M16负收益变为Q64正收益，也提示probe与评分敏感性。ΔN/ΔZ、未选页净残差、I符号准确率、收益排序、regret和全部动作身份分别见page_metrics.csv、performance.csv、selected.csv。','', '## B：动作覆盖与缺失当前信息','', 'H/C表示历史/当前；箭头是移出→加入。以下为M16评分，Q64真实评价。','', '| 事件 | 动作空间 | 真实Oracle收益 | Full M2收益 | History-only收益 |','|---|---|---:|---:|---:|']
 for e in EVENTS:
  for group in ['ALL','HH','HC','CH','CC']:
   def b(m):return next(x for x in life if x['event_id']==e and x['method']==m and x['group']==group and x['probes']==16)
   x,y=b('full_M2'),b('history_only');lines.append(f"| {e.split('_s101')[0]} | {group} | {x['oracle_relative_pct']:+.4f}% | {x['actual_relative_pct']:+.4f}% | {y['actual_relative_pct']:+.4f}% |")
 lines+=['','两例全空间真实最佳动作都属于当前→历史。限制HH后，City Oracle从17.56%降至9.69%，Horse从9.20%降至6.29%，说明动作覆盖确有代价；但HH本身仍有显著收益。','', 'Horse：Full M2与history-only在HH内均获得4.5653%，而history-only放开动作后只获得0.2211%。该例并不是缺少当前估计必然破坏历史之间排序；主要可观察失败是全空间选择了质量较差的当前→历史动作。City：HH中Full M2为4.2006%，history-only为2.2047%，显示另有评分损失。不能把两例归结为同一种原因。','', '附加诊断分别仅将当前残差估计置零、或仅将当前质量修正置零，保留另一部分完整M2；这是教师控制，不是部署方法。两种控制及联合作用详见lifecycle.csv。残差与分母共同决定动作，效果不可简单相加。history-only依旧允许当前动作，当前未增统计，而真实评价始终使用原始当前KV。','', '## 成本与停止决定','', '| K维 | V维 | BF16假设字节/页 | 29层历史缓存GiB | M16主要收缩GFLOPs/层 |','|---|---|---:|---:|---:|']
 for c in costs:lines.append(f"| {c['key_rank']} | {c['value_rank']} | {c['bytes_per_page']} | {c['history_GiB_29layers']:.3f} | {c['M16_contraction_GFLOPs']:.2f} |")
 lines+=['','成本沿用E20对称上三角矩与完整a0公式，不含原KV、原摘要、基、对齐、affine/exp、候选交换及访存等成本。完整V仅用于隔离Key误差；V8行是未进行联合质量验证的成本假设，不能继承fullV质量。rank16/V8额外11968字节/页，超过E20的7KiB探索上限；完整V更昂贵。没有实际延迟或加速比结论。','', '停止：当前基与唯一rank16均未满足质量门槛，且更高维成本不具吸引力，故不继续二阶矩rank搜索、缓存或kernel实现。保留历史候选限制作为诊断发现，不将两例中的HH结果直接升级为线上策略。后续若转向分组或摘要构建，应另行设计实验，而非继续补本轮配置。','', '## 核验与复现','', f'完整B面板共29520个动作收益经独立FP64残差目标重算；A选中动作count/own和真实目标通过；bankQR8评分与E20一致。{len(orig)}个冻结文件哈希未变，36训练事件来源和Query特征谱通过核验。详见audit.json。','', '源码：local_experiments/e21_key_diagnostic/。执行run --setup；run --event <id> --device cuda:0 两例；按预登记触发后run --rank16；最后report。protocol.json、basis_lock.json与rank16_trigger.json记录范围及选择。精简包只包含源码、报告、表、固定基与审计，原始QKV和逐动作大数组留本地。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 pkg=OUT/'review';pkg.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.csv','.md','.json','.npz']:shutil.copy2(p,pkg/p.name)
 shutil.copytree(ROOT/'local_experiments/e21_key_diagnostic',pkg/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(OUT/'e21_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e21_review')
 print('E21 complete',OUT)
if __name__=='__main__':run()
