"""Evidence-only grouping ablation aggregation. Missing runs stay explicitly incomplete."""
import json,csv,statistics,hashlib,datetime,subprocess
from pathlib import Path
SRC=Path(__file__).resolve().parents[1];ROOT=Path('/tmp/group_response_results');OUT=SRC/'artifacts/group_response_178f95a8';OUT.mkdir(exist_ok=True)
GROUPS=['G-K','G-VR','G-KVR','G-ORDER'];events=[json.loads(p.read_text()) for p in sorted((ROOT/'local').glob('*/*.json'))];videos=[json.loads(p.read_text()) for p in sorted((ROOT/'video').glob('*/*/*/result.json'))]
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def table(n,rows):
 if not rows:return
 with (OUT/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
local=[];identity=[]
for g in GROUPS:
 for panel in ['representation','mass','risk','uniform_pool_diagnostic']:
  rs=[e['rows'][g][panel] for e in events]
  if not rs:continue
  local.append(dict(group=g,panel=panel,events=len(rs),nmse=sum(r['squared_error'] for r in rs)/sum(r['target_energy'] for r in rs),event_mean_nmse=statistics.mean(r['nmse'] for r in rs),log_denominator_mae=statistics.mean(r['log_denominator_mae'] for r in rs),denominator_relative_mae=statistics.mean(r['denominator_relative_mae'] for r in rs)))
 for e in events:identity.append(dict(case=e['case'],seed=e['seed'],chunk=e['chunk'],layer=e['layer'],step=e['step'],group=g,**e['rows'][g]['identity']))
table('local_matrix_4x2.csv',[r for r in local if r['panel'] in ['mass','risk']]);table('representation_panel.csv',[r for r in local if r['panel']=='representation']);write('local_aggregate.json',local);write('token_identity_checks.json',identity)
summary=[]
for g in GROUPS:
 for level in ['L2','L3']:
  rs=[r for r in videos if r['group']==g and r['level']==level]
  if rs:summary.append(dict(group=g,level=level,cases=len(rs),latent_nmse=statistics.mean(r['latent_nmse'] for r in rs),**{n:statistics.mean(r['quality'][n] for r in rs) for n in ['lpips','ssim','psnr']}))
table('video_aggregate.csv',summary);table('video_per_case.csv',[dict(case=r['case'],seed=r['seed'],group=r['group'],level=r['level'],latent_nmse=r['latent_nmse'],**r['quality']) for r in videos if r['level']!='Dense'])
costs=[];gates=None
for g in GROUPS:
 rs=[]
 for p in (ROOT/'video').glob('*/'+g+'/Dense/diagnostics.json'):rs+=json.loads(p.read_text())['costs']
 for n in ['group','summary','write']:
  vs=[r['milliseconds'] for r in rs if r['kind']==n and r['chunk']>0]
  if vs:costs.append(dict(group=g,component=n,warm_calls=len(vs),mean_ms=statistics.mean(vs),median_ms=statistics.median(vs),min_ms=min(vs),max_ms=max(vs)))
table('rollout_instrumentation_cost.csv',costs)
if (ROOT/'paired_group_cost.json').exists():
 paired_cost=json.loads((ROOT/'paired_group_cost.json').read_text());costs=[]
 for g in GROUPS:
  rr=[r for r in paired_cost['rows'] if r['group']==g]
  for component,key in [('group','group_ms'),('summary','summary_ms'),('write','write_ms'),('commit_total','commit_total_ms')]:
   vs=[t[key] for r in rr for t in r['trials']];costs.append(dict(group=g,component=component,warm_calls=len(vs),mean_ms=statistics.mean(vs),median_ms=statistics.median(vs),min_ms=min(vs),max_ms=max(vs)))
table('group_commit_cost.csv',costs)
for p in (ROOT/'video').glob('*/*/*/gates.json'):
 x=json.loads(p.read_text())
 if gates is None:gates=x
 else:assert x==gates,'Frozen gates differ'
if gates:
 gs=[]
 for l,values in gates.items():
  for h,row in enumerate(values):
   for f,x in enumerate(row):gs.append(dict(layer=int(l),head=h,feature=f,gate=x))
 table('gates.csv',gs);vs=sorted(r['gate'] for r in gs);write('gate_distribution.json',dict(count=len(vs),min=min(vs),max=max(vs),mean=statistics.mean(vs),mean_abs=statistics.mean(abs(x) for x in vs),std=statistics.pstdev(vs),quantiles={str(q):vs[round(q*(len(vs)-1))] for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},changed=False))
variance=[dict(case=e['case'],seed=e['seed'],chunk=e['chunk'],layer=e['layer'],step=e['step'],group=g,**e['rows'][g]['variance']) for e in events for g in GROUPS]
table('page_variance.csv',variance)
case_local=[]
for case in ['horse_field','traffic_crossing']:
 for seed in [101,303]:
  es=[e for e in events if e['case']==case and e['seed']==seed]
  for g in GROUPS:
   for panel in ['representation','mass','risk']:
    if es:case_local.append(dict(case=case,seed=seed,group=g,panel=panel,events=len(es),nmse=sum(e['rows'][g][panel]['squared_error'] for e in es)/sum(e['rows'][g][panel]['target_energy'] for e in es)))
table('local_per_case.csv',case_local)
paired={}
for panel in ['representation','mass','risk']:
 paired[panel]={}
 for single in ['G-K','G-VR']:
  vals=[e['rows']['G-KVR'][panel]['nmse']-e['rows'][single][panel]['nmse'] for e in events]
  if vals:paired[panel][single]=dict(events=len(vals),joint_win_fraction=sum(x<0 for x in vals)/len(vals),mean_paired_nmse_delta=statistics.mean(vals),median_paired_nmse_delta=statistics.median(vals))
write('joint_paired_local.json',paired)
lookup={(r['group'],r['panel']):r for r in local};decision=dict(status='pending',required=None)
if len(events)==120:
 best=min(['G-K','G-VR'],key=lambda g:lookup[g,'risk']['nmse']);tests={}
 for g in [best,'G-KVR']:
  ratio=lookup[g,'uniform_pool_diagnostic']['nmse']/lookup[g,'representation']['nmse'];wins=sum(e['rows'][g]['uniform_pool_diagnostic']['nmse']<e['rows'][g]['representation']['nmse'] for e in events)/120
  tests[g]=dict(uniform_to_learned_ratio=ratio,uniform_win_fraction=wins,triggered=ratio<.95 and wins>=.6)
 decision=dict(status='evaluated',strongest_single_by_frozen_risk=best,tests=tests,required=any(t['triggered'] for t in tests.values()),budget_per_candidate=1392,criterion='uniform history pooling improves energy-weighted representation NMSE >5% and >=60% of paired events; auxiliary diagnostic only')
write('adaptation_decision.json',decision)
expected={(c,s,g,l) for c in ['horse_field','traffic_crossing'] for s in [101,303] for g in GROUPS for l in ['Dense','L2','L3']};got={(r['case'],r['seed'],r['group'],r['level']) for r in videos};missing=sorted(expected-got)
checks=dict(local_events=len(events),expected_local_events=120,videos=len(videos),expected_videos=48,missing_videos=missing,identities_checked=len(identity),expected_identities=480,token_checks=all(r['token_bijection'] and r['token_kv_exact'] and r['partial_unchanged'] for r in identity),dense_permutation_max_nmse=max((r['dense_nmse'] for r in identity),default=None),frozen_complete=len(events)==120 and not missing,adaptation=decision)
write('audit.json',checks)
lines=['K-response / V-response 历史clean KV分组实验','', '基线178f95a8、summary_pretrained；512×768；338历史页+72当前页；Page64/R4/rank8/M16；固定10%精确预算；全部参数冻结。','G-K/G-VR/G-KVR仅改变完整clean chunk的物理分组特征，原pool_feature_fn不变。G-VR仍依赖K产生attention权重，不是纯V。G-ORDER保持原始token顺序。','前三组保留原平衡划分、页内划分及Config.page_order；G-ORDER无排序、连续64/16切分。当前KV不重排；layer0沿用原提交360历史页；clean全部Dense。','局部120事件：两prompt×两seed×chunk2/5/8×layer1/8/15/22/29×step0/2；每事件按指定公式两个tile、每tile全部64Queries、全部12heads。','局部仅重分完整存活chunk，最老部分chunk原样保留；每组在自己的页面上独立运行mass/risk，不共享跨分区page编号。','闭环L2仅layer15稀疏，L3为layer1–29；9chunk/105帧；候选自身轨迹重新生成后续mask。每组有自己的同协议Dense参照。','Dense参照之间会因浮点累积和真实338页淘汰集合差异而变化；dense_cross_group_differences.json量化了这种差异。视频指标衡量各组对匹配Dense的接近程度，不能直接当作统一目标下的绝对画质排名。','',f'当前完成：局部{len(events)}/120，视频{len(videos)}/48（16个Dense参照+32个L2/L3）。','', '正常路由4×2：能量加权NMSE / log分母MAE']
for g in GROUPS:
 lines.append(g+': '+', '.join(f"{p} {lookup[g,p]['nmse']:.8f} / {lookup[g,p]['log_denominator_mae']:.6f}" for p in ['mass','risk'] if (g,p) in lookup))
lines+=['','表示面板（仅own当前页精确；诊断，不混入10%结果）：']
for r in local:
 if r['panel']=='representation':lines.append(f"{r['group']}: NMSE={r['nmse']:.8f}, log分母MAE={r['log_denominator_mae']:.6f}")
if events:
 one=[e['rows']['G-KVR']['identity'] for e in events];lines+=['',f"实际重分比例：按历史页加权{sum(r['regrouped_pages'] for r in one)/sum(r['history_pages'] for r in one):.6%}；事件平均{statistics.mean(r['regroup_fraction'] for r in one):.6%}。"]
 for chunk in [2,5,8]:
  rr=[r for e,r in zip(events,one) if e['chunk']==chunk]
  if rr:lines.append(f"chunk{chunk}: 历史{sorted(set(r['history_pages'] for r in rr))}页，重分{sorted(set(r['regrouped_pages'] for r in rr))}页，保留部分chunk {sorted(set(r['partial_pages'] for r in rr))}页。")
lines+=['','L2/L3视频（4case均值；RGB[21,101)，LPIPS/SSIM256×384，PSNR512×768）：']
for r in summary:lines.append(f"{r['group']} {r['level']} n={r['cases']}: latent NMSE {r['latent_nmse']:.8f}, LPIPS {r['lpips']:.6f}, SSIM {r['ssim']:.6f}, PSNR {r['psnr']:.3f}")
lines+=['','分组/commit GPU事件成本（最终表：同一真实clean chunk5、layer1/15/29；3次预热后10次交错测量；G-K跳过未使用的V分支）：']
for r in costs:lines.append(f"{r['group']} {r['component']}: mean {r['mean_ms']:.4f} ms, median {r['median_ms']:.4f} ms, n={r['warm_calls']}")
lines+=['原闭环中的原始插桩成本另外保存在rollout_instrumentation_cost.csv；K单支当时计算了未使用的V中间量，该插桩不用于判断单支预处理效率。最终公平成本见paired_group_cost.json。','commit总成本=group+summary+write；单位为每层每个完整clean chunk。包含原参考实现开销，不宣称融合kernel速度；完整运行还包含token身份断言，generation_seconds不可作为纯部署性能。','', '适配判定：'+json.dumps(decision,ensure_ascii=False),'适配未运行。若required=true，则本任务尚未完成，必须追加指定预算适配和对应复测。' if decision['required'] is not False else '未发现上述预先记录的pool不适配信号，故本轮未运行条件性适配；不声称适配无收益。','', '主表为冻结结果；适配结果不得覆盖或混入。token_identity_checks.json记录真实事件的逐候选检查；preflight.json另有原始G-KVR逐位一致与合成边界检查。']
(OUT/'REPORT.zh.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(checks,ensure_ascii=False,indent=2))
