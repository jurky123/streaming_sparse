from .run import *
from .report import table
import tarfile,hashlib

def main():
 gate=json.loads((DEST/'A_gate.json').read_text());assert not gate['passed'];audit=json.loads((DEST/'audit.json').read_text());concentration=[]
 for seed in [17,29]:
  d=json.loads((DEST/f's{seed}/decomposition.json').read_text());event=collections.defaultdict(float);layer=collections.defaultdict(float)
  for x in d:event[x['event']]+=x['total_sse'];layer[x['layer']]+=x['total_sse']
  total=sum(event.values());o=sorted(event.items(),key=lambda x:-x[1]);concentration.append(dict(seed=seed,total_sse=total,layer29_fraction=layer[29]/total,top1_event_fraction=o[0][1]/total,top5_events_fraction=sum(v for k,v in o[:5])/total,top1_event=o[0][0]))
 table('error_concentration.csv',concentration)
 costs=[dict(method='Shared-Q',scalars_per_probe=1,permanent_page_bytes=0,extra_raw_kv_bytes=0,scalar_add_GFLOPs_per_layer=16*410*864/1e9,online_calibrator_trained=False),dict(method='CandidateBackground-Q',scalars_per_probe=2,permanent_page_bytes=0,extra_raw_kv_bytes=0,scalar_add_GFLOPs_per_layer=16*410*864/1e9,online_calibrator_trained=False)];table('cost_estimate.csv',costs)
 p=DEST/'REPORT.md';text=p.read_text().split('\n\n## 主要判断')[0];text+='''

## 主要判断

A2双偏移确有部分信息价值：两种子的实际Set恢复均提高，并且均超过同信息Independent；但只恢复E29→TrueBeta缺口的31.56%和11.09%，未达到50%。A1单偏移则连跨种子的收益改善都不稳定。停止当前共享校准，不训练B，不增加组数或改权重。

可靠性仍不足。A2退化数从95/91降至83/80，P10改善，但逐例平均仍为−3.31%/−3.56%。Seed29最差绝对NMSE增量和逐例平均比E29更差，因此不能只凭汇总正收益或较少错误次数判定成功。

回归和决策再次分离：双偏移解释的汇总mass²误差约99.68%/99.73%，事件/Query等权则为64.62%/62.92%。前者并不表示约99.7%的页面误差或组合信息都已恢复；层级和集中度如下，完整分组见quality_summary.csv、generalization.csv。
'''
 for x in concentration:text+=f"\n- seed{x['seed']}：layer29占原加权误差{x['layer29_fraction']:.2%}；最大单事件占{x['top1_event_fraction']:.2%}，前五事件占{x['top5_events_fraction']:.2%}。\n"
 text+='''
A2的条件指标也不一致：seed29汇总双重中心化相关升至0.598，但逐事件条件相关中位数仅从0.455升至0.476，符号翻转召回从8.91%降至8.75%；seed17双重中心化相关仅0.167。因此不能单凭一个较高的汇总相关判定已获得可靠条件信息。

这些结果不支持继续用少数Query共享平移补偿当前Reader。剩余页面差异、预测局部响应与质量修正的耦合仍可能影响动作；不能仅凭回归教师失败断言所有决策相关β误差都是页面特异的，更不能证明R4信息论不足。此实验没有求共享偏移的决策最优解。

审计包括576事件的独立NumPy longdouble加权偏移拟合，抽样独立正混合集合评分与Independent全512分数、固定mask/count/own约束，以及E29/TrueBeta两端原始动作重放。所有源缓存与冻结模型哈希保持一致；没有读取raw KV文件或访问Test。审计细节见audit.json、source_hashes.csv。

轻量评审包包含报告、CSV、协议、审计与源代码；完整逐动作和五状态面板保留在s17/s29目录。此轮没有可部署的新校准器，也没有实测新增延迟。
''';p.write_text(text)
 (ROOT/'local_experiments/e37_shared_beta/README.md').write_text('''# E37-A shared beta diagnostics

Frozen E29/E30 cache only, same M16 Risk and Medium512. mass-squared LS intercepts per query; one all-page intercept or candidate/background pair. Teacher-only, no training/Test/raw KV loading.

Run `python -m local_experiments.e37_shared_beta.run --seed 17` and seed29, then `report`, `audit`, `finish` modules in that order. Completed runs are protected from overwrite. A failed; B is not started. Protocol in artifacts/e37_shared_beta_v1/protocol.json. Full per-action panels are kept in s17/s29; the review tar contains tables and audit/source.
''')
 src=list(csv.DictReader((DEST/'source_hashes.csv').open()));h={x['path']:x['sha256'] for x in src}
 for f in (ROOT/'local_experiments/e37_shared_beta').iterdir():
  if f.suffix in ['.py','.md']:h[str(f.relative_to(ROOT))]=sha(f)
 table('source_hashes.csv',[dict(path=k,sha256=v) for k,v in sorted(h.items())]);items=sorted([f for f in DEST.iterdir() if f.suffix in ['.json','.csv','.md'] and f.name!='package_manifest.json']+list((ROOT/'local_experiments/e37_shared_beta').glob('*.py'))+[ROOT/'local_experiments/e37_shared_beta/README.md']);manifest=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in items];dump(DEST/'package_manifest.json',manifest)
 with tarfile.open(DEST/'e37_a_review.tar','w:gz') as tar:
  for f in items+[DEST/'package_manifest.json']:tar.add(f,arcname=str(f.relative_to(ROOT)))
 with tarfile.open(DEST/'e37_a_review.tar','r:gz') as tar:
  for x in manifest:assert hashlib.sha256(tar.extractfile(x['path']).read()).hexdigest()==x['sha256']
 print('PACKAGE VERIFIED',len(items),(DEST/'e37_a_review.tar').stat().st_size,concentration)
if __name__=='__main__':main()
