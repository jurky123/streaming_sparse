"""Report factorial grouping experiments without blending inner-feature effects."""
import json,csv,statistics,math
from pathlib import Path
SRC=Path(__file__).resolve().parents[1];ROOT=Path('/tmp/group_hierarchy_results');OUT=SRC/'artifacts/group_hierarchy_178f95a8';OUT.mkdir(exist_ok=True)
GROUPS=['G-ORDER','G-K','G-K-joint','G-KVR','G-K3-V2','G-K3-V2-Kinner'];LAYERS=[1,8,15,22,29];CASES=[f'{c}_s{s}' for c in ['horse_field','traffic_crossing'] for s in [101,303]]
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def table(n,rows):
 if not rows:return
 with (OUT/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
events=[json.loads(p.read_text()) for p in sorted((ROOT/'local').glob('*/*.json'))];local=[];identities=[];variance=[]
for g in GROUPS:
 for panel in ['representation','mass','risk']:
  rs=[e['rows'][g][panel] for e in events]
  if not rs:continue
  weights=[e['history_pages']+72 for e in events]
  local.append(dict(group=g,panel=panel,events=len(rs),nmse=sum(r['squared_error'] for r in rs)/sum(r['target_energy'] for r in rs),event_mean_nmse=statistics.mean(r['nmse'] for r in rs),page_log_denominator_mae=sum(r['page_log_denominator_mae']*w for r,w in zip(rs,weights))/sum(weights),event_mean_page_log_denominator_mae=statistics.mean(r['page_log_denominator_mae'] for r in rs),history_page_log_denominator_mae=sum(r['history_page_log_denominator_mae']*e['history_pages'] for r,e in zip(rs,events))/sum(e['history_pages'] for e in events),global_log_denominator_mae=statistics.mean(r['log_denominator_mae'] for r in rs)))
 for e in events:
  identities.append(dict(case=e['case'],seed=e['seed'],chunk=e['chunk'],layer=e['layer'],step=e['step'],group=g,**e['rows'][g]['identity']))
  variance.append(dict(case=e['case'],seed=e['seed'],chunk=e['chunk'],layer=e['layer'],step=e['step'],group=g,**e['rows'][g]['variance']))
table('local_three_panels.csv',local);write('local_aggregate.json',local);write('token_identity_checks.json',identities);table('page_variance.csv',variance)
paired=[]
for inner,baseline,new in [('K','G-K','G-K3-V2-Kinner'),('joint','G-K-joint','G-K3-V2')]:
 for panel in ['representation','mass','risk']:
  for e in events:
   a=e['rows'][new][panel];b=e['rows'][baseline][panel];paired.append(dict(inner=inner,baseline=baseline,candidate=new,panel=panel,case=e['case'],seed=e['seed'],chunk=e['chunk'],layer=e['layer'],step=e['step'],baseline_nmse=b['nmse'],candidate_nmse=a['nmse'],nmse_delta=a['nmse']-b['nmse'],page_log_denominator_delta=a['page_log_denominator_mae']-b['page_log_denominator_mae']))
table('paired_local.csv',paired)
videos=[json.loads(p.read_text()) for p in sorted((ROOT/'video').glob('*/*/*/result.json')) if p.parent.parent.name in GROUPS];vr=[];va=[]
for x in videos:
 if x['level']=='Dense':continue
 vr.append(dict(case=x['case'],seed=x['seed'],group=x['group'],level=x['level'],latent_nmse=x['latent_nmse'],**x['quality']))
for g in GROUPS:
 for level in ['L2_'+str(l) for l in LAYERS]+['L3']:
  rs=[r for r in vr if r['group']==g and r['level']==level]
  if rs:va.append(dict(group=g,level=level,cases=len(rs),**{n:statistics.mean(r[n] for r in rs) for n in ['latent_nmse','lpips','ssim','psnr']}))
table('video_per_case.csv',vr);table('video_by_layer.csv',va)
means=[]
for g in GROUPS:
 for level in ['L2','L3']:
  rs=[r for r in vr if r['group']==g and (r['level'].startswith('L2_') if level=='L2' else r['level']=='L3')]
  if rs:means.append(dict(group=g,level=level,units=len(rs),**{n:statistics.mean(r[n] for r in rs) for n in ['latent_nmse','lpips','ssim','psnr']}))
table('video_aggregate.csv',means)
cost=[]
if (ROOT/'paired_group_cost.json').exists():
 raw=json.loads((ROOT/'paired_group_cost.json').read_text());table('cost_by_layer.csv',[{k:v for k,v in r.items() if k!='trials'} for r in raw['rows']])
 for g in GROUPS:
  rr=[r for r in raw['rows'] if r['group']==g];ts=[t for r in rr for t in r['trials']]
  cost.append(dict(group=g,trials=len(ts),**{n:statistics.mean(t[n] for t in ts) for n in ['group_ms','summary_ms','write_ms','commit_total_ms']}))
table('group_commit_cost.csv',cost)
restored=[json.loads(p.read_text()) for p in (ROOT/'restoration').glob('*.json')];status=dict(local_events=len(events),expected_local_events=120,video_results=len(videos),expected_video_results=168,reused_video_results=sum('reused_from' in r for r in videos),restored_cases=len(restored),restoration_bitwise_pass=len(restored)==4 and all(r['latent_bitwise_equal'] for r in restored),complete=len(events)==120 and len(videos)==168 and len(restored)==4 and len(cost)==6)
write('status.json',status)
lines=['分层 K3→V2 分组实验：冻结权重，双页内方案','',f'状态：局部{len(events)}/120，视频{len(videos)}/168，基线恢复{len(restored)}/4。','基线178f95a8/summary_pretrained；338历史页，Page64/R4/rank8/M16，固定10%，原current KV及clean Dense，所有参数冻结、不训练。','G-K：五级均K，旧K页内；G-K-joint：五级均K，联合页内；G-KVR：五级及页内均原联合；G-K3-V2：外层KKKVV、联合页内；G-K3-V2-Kinner：外层KKKVV、K页内；G-ORDER：原序64/16，不排序。','新方案在整个4608-token chunk上各计算一次K/V响应；不在256-token父组内重算。前三层18个父组保持K划分，后两层在父组内用V排序，最后按各自指定特征做页内[2,2]和原Config.page_order。','必须配对比较：G-K3-V2 vs G-K-joint（联合页内），G-K3-V2-Kinner vs G-K（K页内）；不要将跨页内方案差异归因于外层最后两级。','上一轮未存原QKV，只存指标及latent。按同事件索引确定性恢复；每个case最终latent与旧G-KVR L3逐位相同，原对照指标匹配；原QKV已补存并SHA256校验。没有重新抽样。','layer0按指定提交保持Dense及360历史页；稀疏层历史338页。精确预算按原代码floor(0.1*总页数)，包含own页：chunk2为21页，chunk5/8为41页。','局部仅重分完整存活chunk，最老部分chunk原页与token不变；338页稳态288页重分，50页保留。每个候选在自己的页面上选mass/risk，不复用跨分区page ID。','', '局部主表：NMSE=sum平方误差/sum Dense输出能量；page logZ MAE按所有head/query/page/事件加权，所选精确页误差为零。另存历史页、未选页及整体logZ误差；不得把整体logZ MAE称为页级误差。','仅own当前页精确是独立表示诊断，不混入10%混合结果。','group / panel / NMSE / page logZ MAE']
for r in local:lines.append(f"{r['group']} / {r['panel']} / {r['nmse']:.8f} / {r['page_log_denominator_mae']:.6f}")
lines+=['','闭环：L2单层1/8/15/22/29逐一稀疏，L3层1–29；两prompt×两seed、9chunk/105帧，各自轨迹更新mask，匹配各组Dense。L2均值包含20单元，L3包含4单元；逐层与逐case表另附。','LPIPS/SSIM计算RGB[21,101)的256×384版本，PSNR原512×768；latent NMSE包含全部27 latent帧。各组Dense会因自身分组/淘汰/数值累积而不同；这些是近似一致性指标，不是绝对画质排名。']
for r in means:lines.append(f"{r['group']} {r['level']} n={r['units']}: latent NMSE={r['latent_nmse']:.6f}, LPIPS={r['lpips']:.6f}, SSIM={r['ssim']:.6f}, PSNR={r['psnr']:.3f}")
lines+=['','公平成本：相同真实horse_field seed101 clean chunk5、layer1/15/29；3次预热+10次交错；每层每chunk GPU事件毫秒。']
for r in cost:lines.append(f"{r['group']}: group {r['group_ms']:.4f} + summary {r['summary_ms']:.4f} + write {r['write_ms']:.4f} = commit {r['commit_total_ms']:.4f} ms")
lines+=['','不增加训练，未运行任何适配。未完成项目以status.json明确计数，不将待运行项记成完成。']
(OUT/'REPORT.zh.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(status,indent=2))
