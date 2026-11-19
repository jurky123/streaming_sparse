"""Pair conclusions within each inner feature; never blend different controls."""
import json,csv,statistics,hashlib,datetime
from pathlib import Path
SRC=Path(__file__).resolve().parents[1];OUT=SRC/'artifacts/group_hierarchy_178f95a8';ROOT=Path('/tmp/group_hierarchy_results')
def load(p):return json.loads(p.read_text())
assert load(OUT/'final_audit.json')['passed']
local=load(OUT/'local_aggregate.json');li={(r['group'],r['panel']):r for r in local}
video=list(csv.DictReader((OUT/'video_aggregate.csv').open()));vi={(r['group'],r['level']):r for r in video};bylayer=list(csv.DictReader((OUT/'video_by_layer.csv').open()));bli={(r['group'],r['level']):r for r in bylayer};cost=list(csv.DictReader((OUT/'group_commit_cost.csv').open()));ci={r['group']:r for r in cost}
rows=[];lines=['分层 K3→V2 实验结论','', '结论：本次冻结权重、整chunk响应复用的K3→V2方案，没有优于同页内规则的全K外层。局部表示、mass和risk三面板都更差；两套页内规则下，L3平均近似一致性也更差。不建议替换当前全K外层或原顺序基线。','', '两套严格配对：']
for inner,base,new in [('联合','G-K-joint','G-K3-V2'),('K','G-K','G-K3-V2-Kinner')]:
 lines+=['',f'{inner}页内：{new} 对 {base}']
 for panel in ['representation','mass','risk']:
  a,b=li[new,panel],li[base,panel];rel=a['nmse']/b['nmse']-1;row=dict(inner=inner,setting=panel,baseline=base,candidate=new,baseline_nmse=b['nmse'],candidate_nmse=a['nmse'],relative_change=rel,baseline_page_log_mae=b['page_log_denominator_mae'],candidate_page_log_mae=a['page_log_denominator_mae']);rows.append(row)
  lines.append(f"  {panel}: NMSE {b['nmse']:.8f} → {a['nmse']:.8f}（{rel:+.2%}）；页级log分母MAE {b['page_log_denominator_mae']:.6f} → {a['page_log_denominator_mae']:.6f}。")
 for level in ['L2','L3']:
  a,b=vi[new,level],vi[base,level];lines.append(f"  {level}: latent NMSE {float(b['latent_nmse']):.6f} → {float(a['latent_nmse']):.6f}；LPIPS {float(b['lpips']):.6f} → {float(a['lpips']):.6f}；SSIM {float(b['ssim']):.6f} → {float(a['ssim']):.6f}；PSNR {float(b['psnr']):.3f} → {float(a['psnr']):.3f}。")
 for l in [1,8,15,22,29]:
  a,b=bli[new,'L2_'+str(l)],bli[base,'L2_'+str(l)];lines.append(f"    L2 layer{l}: LPIPS {float(b['lpips']):.6f} → {float(a['lpips']):.6f}；latent NMSE {float(b['latent_nmse']):.6f} → {float(a['latent_nmse']):.6f}。");rows.append(dict(inner=inner,setting='L2_'+str(l),baseline=base,candidate=new,baseline_nmse=float(b['latent_nmse']),candidate_nmse=float(a['latent_nmse']),baseline_lpips=float(b['lpips']),candidate_lpips=float(a['lpips'])))
 a,b=ci[new],ci[base];lines.append(f"  commit {float(b['commit_total_ms']):.4f} → {float(a['commit_total_ms']):.4f} ms/层/chunk，group {float(b['group_ms']):.4f} → {float(a['group_ms']):.4f} ms。小于1%的时间差不作为确定的性能收益。")
lines+=['','关键解释与边界：','1. risk面板新方案仅在5/120（联合页内）和4/120（K页内）事件获胜。两种页内规则给出一致的局部负面结果，故不能把退步归因于页内特征混杂。','2. 新方案历史页内K方差约1.356，全K约1.149；V方差约0.621，全K约0.610。就这些原始方差指标看，后两级V-response没有使页面更紧凑。这是解释性观测，不能单凭方差建立因果关系。',f"3. G-ORDER risk混合NMSE为{li['G-ORDER','risk']['nmse']:.8f}；新方案约为其{li['G-K3-V2','risk']['nmse']/li['G-ORDER','risk']['nmse']:.2f}倍，仍明显落后于原始顺序。G-ORDER L3 LPIPS为{float(vi['G-ORDER','L3']['lpips']):.6f}，commit约{float(ci['G-ORDER']['commit_total_ms']):.4f} ms。",'4. G-ORDER的未加注意力质量权重的页级logZ MAE较高，但混合输出NMSE更低。页级分母误差不是最终混合输出误差的替代；原始全局logZ MAE另行记录。','L2例外必须保留：联合页内layer15的LPIPS由0.121265降至0.119806；K页内layer22由0.078840降至0.073694。不是每个层都退步，但总体L2和L3均未受益。','5. L2平均覆盖5层×4case=20单元，不能代替逐层结果。L3为4case。所有视频指标都对各组匹配Dense，反映近似一致性，不等于统一参考下的绝对视觉质量排名。','6. 本次只检验整chunk计算并复用响应；不在256-token组内重算V-response。不训练、不适配，不外推到其他gate、训练方案或局部响应重算。','', '执行与完整性：','指定提交178f95a8、summary_pretrained；原算法与参数冻结；338历史页、Page64/R4/rank8/M16、10%固定预算，当前KV不重排、clean Dense。','120个旧事件恢复完成，四个case的完整baseline latent与上一轮逐位相同；恢复的旧控制指标最大绝对差为0。上一轮原始QKV未落盘，不能直接声称原QKV文件逐字节相同；本轮恢复的固定QKV已保存且SHA256校验，所有六条件使用相同文件。','局部720组候选token身份检查通过；完整chunk重分、最老50页不动。预验证证明KKKKK/VVVVV的分阶段实现与原balanced一致，KKKVV与原balanced按3级+2级组合实现一致；更换页内特征不改变page token集合。','168条视频记录全部完成：24个匹配Dense+120个五层L2+24个L3。36条兼容结果复用、132条新运行。全部9chunk/105帧，RGB质量窗口[21,101)。','同一真实clean chunk5、layer1/15/29上预热3次、交错10次，记录group+summary+write=commit。','没有未完成的协议项目。训练/适配明确未运行。','', '主要文件：REPORT.zh.txt；local_three_panels.csv；video_by_layer.csv；video_per_case.csv；group_commit_cost.csv；final_audit.json；qkv/原始输入；video/全部视频；L2_layers.png；L3_frame61.png。']
(OUT/'CONCLUSIONS.zh.txt').write_text('\n'.join(lines)+'\n');(OUT/'paired_summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
p=OUT/'protocol.json';x=load(p);x.update(status='complete',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
paths=list((SRC/'local_experiments/group_hierarchy').glob('*.py'))+list((SRC/'local_setup').glob('*group_hierarchy.py'));(OUT/'code_manifest.json').write_text(json.dumps({str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print('\n'.join(lines[:12]))
