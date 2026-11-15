from .report import *
from ..e24_joint_representatives.finalize import stats

def finish():
 assert all((E25/f'D_s{s}'/'complete.json').exists() for s in [17,29])
 training=[];curve=[];checks=[]
 for seed in [17,29]:
  dest=E25/f'D_s{seed}';done=json.loads((dest/'training_complete.json').read_text());training.append(done);ck=torch.load(dest/'selected.pt',map_location='cpu',weights_only=False);source=torch.load(checkpoint('C',seed),map_location='cpu',weights_only=False);assert all(torch.equal(v,source['layers'][n]) for n,v in ck['layers'].items())
  crows=json.loads((E25/f'C_s{seed}'/'ladder.json').read_text());drows=json.loads((dest/'ladder.json').read_text());cr={x['event']:x for x in crows if x['method']=='P0'}
  teacher_ref={(r['event'],r['method']):r for r in crows}
  for r in drows:
   if r['method']!='P0':
    ref=teacher_ref[(r['event'],r['method'])];assert r['remove']==ref['remove'] and r['add']==ref['add'] and r['execute']==ref['execute'] and abs(r['executed_gain']-ref['executed_gain'])<1e-12
   assert abs(r['initial']-cr[r['event']]['initial'])<1e-12 and abs(r['oracle_gain']-cr[r['event']]['oracle_gain'])<1e-12
  for ep in range(4):checks.append(dict(seed=seed,epoch=ep,selected=ep==done['best']['epoch'],**stats(json.loads((dest/f'dev_epoch{ep}.json').read_text()))))
  curve.extend(dict(seed=seed,**r) for r in json.loads((dest/'curves.json').read_text()))
 dump(E25/'retraining_audit.json',dict(runs=training,all_frozen_C_tensors_exact=True,new_initial_and_oracle_equal_C=True,test_evaluated=False))
 table('reader_checkpoints.csv',checks);table('reader_training_curves.csv',curve)
 lossstats=[]
 for seed in [17,29]:
  vals=[x for x in curve if x['seed']==seed]
  for key in ['residual','set_loss','total']:
   v=np.array([x[key] for x in vals]);assert np.isfinite(v).all();lossstats.append(dict(seed=seed,loss=key,median=float(np.median(v)),p99=float(np.quantile(v,.99)),max=float(v.max()),fraction_over_1e6=float((v>1e6).mean())))
 table('reader_loss_dynamic_range.csv',lossstats)
 summary=list(csv.DictReader((E25/'ladder_summary.csv').open()));cond=list(csv.DictReader((E25/'conditional_summary.csv').open()));brief=[]
 for group in ['A','C','D']:
  for seed in [17,29]:
   s={r['method']:r for r in summary if r['group']==group and int(r['seed'])==seed};c=next(r for r in cond if r['group']==group and int(r['seed'])==seed);brief.append(dict(group=group,seed=seed,R_teacher16=s['P1']['R_executed'],R_proposal=s['P0']['R_proposal'],R_executed=s['P0']['R_executed'],R_conditional=c['R_conditional_raw']))
 table('four_numbers.csv',brief)
 reportpath=E25/'REPORT.md'
 with reportpath.open('a') as f:
  f.write('\n## 冻结重训结果\n\n|种子|原C执行恢复|D执行恢复|原C事后门控恢复|D事后门控恢复|D逐例平均改善|D汇总改善|\n|---|---:|---:|---:|---:|---:|---:|\n')
  for seed in [17,29]:
   c=next(r for r in summary if r['group']=='C' and int(r['seed'])==seed and r['method']=='P0');d=next(r for r in summary if r['group']=='D' and int(r['seed'])==seed and r['method']=='P0');f.write(f"|{seed}|{float(c['R_executed']):.2%}|{float(d['R_executed']):.2%}|{float(c['R_proposal']):.2%}|{float(d['R_proposal']):.2%}|{float(d['mean_improvement_pct']):+.3f}%|{float(d['pooled_improvement_pct']):+.3f}%|\n")
  f.write('\n## 判断边界\n\n1. 当前固定16维真残差保留了相当一部分决策能力，且P1到P2的差距远小于P0到P1；继续盲目扩大输出维度缺乏依据。\n2. 即使完美过滤模型提议中的负收益，恢复率仍远低于P1，排序与残差坐标预测是独立于门控的瓶颈。\n3. P2到P3仍有明显损失，说明在新摘要下M16的Query覆盖是另一限制；不能继承旧摘要PCA16的恢复率。\n4. 当前条件相关和翻转召回偏低，没有证据表明Joint显著学会了跨集合动作变化。共同面板与E13/E14不同，翻转发生率不能直接横向比较。\n5. D若改善，支持冻结目标可能缓解部分训练困难；但从头初始化、训练目标分布和优化轨迹也发生变化，不能将全部改善因果归于目标漂移。D仍沿用原固定损失尺度，部分事件损失动态范围很大（reader_loss_dynamic_range.csv），优化可预测性仍未被彻底排除。\n6. P1成功证明的是“真实坐标具有决策价值”，不是“R4一定编码了可读出的坐标”。D失败也不能证明摘要没有信息；当前实验仍无法彻底区分可观测性、因子化读出结构和损失优化。Test继续封存，不增加网络或采集数据。\n')
  f.write('\n独立数值审计见independent_audit.json（1005个教师动作）与panel_audit.json（300个条件动作）；原始E24 P0动作完整重现，P3直接原目标一致。D摘要逐tensor不变，初选NMSE和同池Oracle逐例与对应C相同。四个预定epoch均保留，不用Test挑选。\n\n完整逐状态面板位于A_s*/C_s*/D_s*/panel_*.npz；精简包不含这些可重建面板、原始98GB QKV或中间epoch模型，包含选中模型、固定投影基、代码、报告和逐事件/分层CSV。\n')
 pkg=E25/'review';pkg.mkdir(exist_ok=True)
 for p in E25.iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md']:shutil.copy2(p,pkg/p.name)
 shutil.copy2(OUT/'fit.pt',pkg/'fit.pt')
 for group in ['A','C','D']:
  for seed in [17,29]:shutil.copy2(checkpoint(group,seed),pkg/f'{group}_s{seed}_selected.pt')
 shutil.copytree(ROOT/'local_experiments/e25_decision_diagnostics',pkg/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 # Include imported E24 machinery for traceability, while retaining repository imports.
 shutil.copytree(ROOT/'local_experiments/e24_joint_representatives',pkg/'e24_source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(E25/'provenance.json',dict(code={p.name:sha(p) for p in (ROOT/'local_experiments/e25_decision_diagnostics').glob('*.py')},models={f'{g}_{s}':sha(checkpoint(g,s)) for g in ['A','C','D'] for s in [17,29]},protocol_sha=sha(E25/'protocol.json'),E24_manifest_sha=sha(OUT/'dataset_manifest.json'),fit_sha=sha(OUT/'fit.pt'),test_evaluated=False));shutil.copy2(E25/'provenance.json',pkg/'provenance.json');dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(E25/'e25_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e25_review')
 print('PACKAGED',E25/'e25_review.tar.gz')
if __name__=='__main__':finish()
