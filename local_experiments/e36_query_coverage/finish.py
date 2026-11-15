from .run import *
from .report import table
import csv,tarfile,hashlib

def main():
 old=list(csv.DictReader((ROOT/'artifacts/e35_candidate_recall_v1/topL_per_event.csv').open()));lookup={(int(x['seed']),x['event'],int(x['L'])):float(x['best_gain']) for x in old if x['policy']=='Set'};e25=list(csv.DictReader((ROOT/'artifacts/e25_decision_diagnostics_v1/ladder_per_event.csv').open()));teachers={(int(x['seed']),x['event'],x['method']):float(x['executed_gain']) for x in e25 if x['group']=='C'};errors=[]
 for seed in [17,29]:
  dest=DEST/f'A_s{seed}'
  for x in json.loads((dest/'topL.json').read_text()):
   if x['method']=='E29-M16':errors.append(abs(x['best_gain']-lookup[seed,x['event'],x['L']]))
  for x in json.loads((dest/'decisions.json').read_text()):
   if x['method'] in ['True-M16','True-Q64']:errors.append(abs(x['executed_gain']-teachers[seed,x['event'],'P2' if x['method']=='True-M16' else 'P3']))
 assert max(errors)<1e-8;dump(DEST/'baseline_replay.json',dict(checks=len(errors),max_absolute_gain_difference=max(errors),E35_topL=True,E25_P2_P3=True))
 summary=list(csv.DictReader((DEST/'decision_summary.csv').open()));get=lambda s,m,p='Set':next(x for x in summary if int(x['seed'])==s and x['method']==m and x['policy']==p)
 costs=[]
 for name,n in [('Uniform16',16),('Predicted-Q64',64)]:costs.append(dict(method=name,probes=n,query_dependent_compute_ratio=n/16,additional_raw_kv_bytes=0,additional_page_cache_bytes=0,risk_probes=16,actions=512,status='executed'))
 costs.append(dict(method='Coreset16',probes=16,query_dependent_compute_ratio=1,additional_raw_kv_bytes=0,additional_page_cache_bytes=0,risk_probes=16,actions=512,selector_distance_GFLOPs_per_layer=3*16*64*128*864/1e9,status='not executed; capacity/cost estimate only'));table('cost_estimate.csv',costs)
 p=DEST/'REPORT.md';text=p.read_text().split('\n\n## 结论与边界')[0];text+='''

## 结论与边界

A未通过，B未执行。当前模型把Query从16扩为64，不能稳定提高真实交换收益或Top8召回；不继续训练Query Selector，也不扩大Query选择配置搜索。这不等于Query覆盖没有损失：真实128维/M16仅恢复60.07%和65.69%，但现有预测器无法稳定利用完整覆盖。

seed17实际恢复增加9.14个百分点，同时Top8下降6.23个百分点；seed29实际恢复下降7.57个百分点，Top8只增加1.25个百分点。两种子的Top1事后真门控均改善，但seed29实际执行反而变差，说明提议质量、错误动作损失与接受判断不能由召回指标互相替代。

轨迹分列显示，seed29的Q64实际恢复在kitchen增加1.66个百分点、在snow下降33.86个百分点；因此不能用单轨迹正面结果支持继续。seed17两条轨迹实际恢复均增加，但其Top8变化方向也不一致。

在相同Q64下，仅把β换成真值，恢复率从20.73%/17.05%提高到65.94%/62.32%；Top8达到80.19%/81.23%。这支持质量预测误差仍是重要限制，但教师替换同时改变误差抵消，不能证明β是唯一原因。TrueBeta-Q64仍有37/288、42/288事件退化，P10均为负；没有达到可靠执行标准。

TrueBeta-Q64相对既有TrueBeta-M16（46.37%/55.48%）的提升表明，在较准质量信息下，扩大Query覆盖确有潜在价值。它不是在线改进方案：既读取教师β，又使用四倍Query评分。下一步应把主要问题定位回局部响应/质量修正的可预测性和决策可靠性，而不是把此次停止解释为否定所有Query采样设计。

## 审计与交付

所有576个事件重放原mask、候选页对与真实收益标签；所有执行满足固定count与own保护。抽取每种子跨轨迹/层事件及最差绝对退化事件，以独立NumPy longdouble正混合公式核验预测评分和真实Q64收益。M16 Top1/4/8逐事件对齐E35，True-M16/True-Q64对齐E25 P2/P3。具体误差、样本数及哈希见audit.json、baseline_replay.json、source_hashes.csv。

CSV保留Set与同信息Independent、完整逐事件和分组指标；Teacher/Top-L结果均明确为离线诊断。未开展B、训练、Test或Kernel。压缩包包含报告、协议、全部CSV、审计及代码；逐动作/跨状态NPZ保留在完整实验目录，避免把大数组混入轻量评审包。
''';p.write_text(text)
 readme='''# E36 Query coverage

Run A with the frozen E29/E30 full64 caches and canonical dev data:

```bash
python -m local_experiments.e36_query_coverage.run --seed 17
python -m local_experiments.e36_query_coverage.run --seed 29
python -m local_experiments.e36_query_coverage.report
python -m local_experiments.e36_query_coverage.audit
python -m local_experiments.e36_query_coverage.finish
```

A gate failed in this run. B is not executed. `--stage B` requires the persisted A gate to pass. Rerunning a completed run requires a separate output directory; no overwrite of complete runs. Full panels are in A_s17/A_s29; compact review tar contains reports/CSVs/audits/source only. Development has 288 events total per model seed, 144 per trajectory. Test remains sealed.
''';(ROOT/'local_experiments/e36_query_coverage/README.md').write_text(readme)
 source_rows=list(csv.DictReader((DEST/'source_hashes.csv').open())); source_map={x['path']:x['sha256'] for x in source_rows}
 for source in (ROOT/'local_experiments/e36_query_coverage').iterdir():
  if source.suffix in ['.py','.md']:source_map[str(source.relative_to(ROOT))]=sha(source)
 table('source_hashes.csv',[dict(path=k,sha256=v) for k,v in sorted(source_map.items())])
 assert (DEST/'audit.json').exists();items=[p for p in DEST.iterdir() if p.suffix in ['.md','.csv','.json'] and p.name!='package_manifest.json']+list((ROOT/'local_experiments/e36_query_coverage').glob('*.py'))+[ROOT/'local_experiments/e36_query_coverage/README.md'];items=sorted(items);manifest=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x),bytes=x.stat().st_size) for x in items];dump(DEST/'package_manifest.json',manifest)
 with tarfile.open(DEST/'e36_a_review.tar','w:gz') as tar:
  for x in items+[DEST/'package_manifest.json']:tar.add(x,arcname=str(x.relative_to(ROOT)))
 with tarfile.open(DEST/'e36_a_review.tar','r:gz') as tar:
  for x in manifest:assert hashlib.sha256(tar.extractfile(x['path']).read()).hexdigest()==x['sha256']
 print('PACKAGE VERIFIED',len(items), 'bytes',(DEST/'e36_a_review.tar').stat().st_size,'replay',max(errors))
if __name__=='__main__':main()
