from .run import *

def main():
 summary=json.loads((DEST/'summary.json').read_text());poolsummary=json.loads((DEST/'pool_summary.json').read_text());gate=json.loads((DEST/'gate.json').read_text());audit=json.loads((DEST/'independent_audit.json').read_text());rows=list(csv.DictReader((DEST/'topL_per_event.csv').open()));general=list(csv.DictReader((DEST/'generalization.csv').open()));base=list(csv.DictReader((DEST/'baseline_per_event.csv').open()));ss={(r['seed'],r['policy'],r['L']):r for r in summary};ps={(r['seed'],r['pool']):r for r in poolsummary}
 lines=['# E35-A：交换候选召回与稀疏验证前的收益上限','', '**结论：Top-8能较常包含正收益动作，但未覆盖预登记要求的大部分高收益动作；停在A，不执行E35-B精确Key验证。**','', '## 范围与评价规则','', '仅重放冻结E29在原Risk初始状态的Medium512个动作分数与真实Q64收益。模型种子17/29各288事件，来自同样两条开发轨迹。没有读取原始QKV、执行Attention、运行Reader、训练、修改β或访问Test。Independent顺序由已保存的预测响应/质量/摘要中心计算，不获取新信息。','', '按预测分数降序取Top1/2/4/8，精确平局用原面板下标；即使预测分数为负也保留候选，避免把召回诊断混入接受门控。每个候选集的“收益”均为事后用真实标签选取最佳正收益或no-op，属于离线上限，不能称为线上执行结果。排序本身不读取标签。','', '恢复率=ΣTop-L最佳真实正收益/Σ原Medium最佳真实正收益。正收益机会与最佳动作命中率以Medium Oracle>1e−12的事件为分母：seed17为278，seed29为280。全288事件口径也在CSV。最佳命中容差1e−12；相对NMSE仅在初始J>1e−12定义。所有绝对汇总保留全部事件。','', '预登记参考门槛：两个种子Set Top8均≥70%恢复、且≥80%正收益机会。这是本轮对“大部分收益”的操作化标准，不是用户给定的精确阈值，也不是线上成功门槛；无论是否通过，本轮只执行A。','', '## Set候选的收益覆盖','', '|L|seed17恢复|seed29恢复|seed17正收益机会|seed29正收益机会|seed17命中最佳动作|seed29命中最佳动作|','|---:|---:|---:|---:|---:|---:|---:|']
 for L in LS:
  a,b=[ss[s,'Set',L] for s in [17,29]];lines.append(f"|{L}|{a['recovery']:.2%}|{b['recovery']:.2%}|{a['positive_coverage']:.2%}|{b['positive_coverage']:.2%}|{a['best_hit_fraction']:.2%}|{b['best_hit_fraction']:.2%}|")
 lines+=['','Top1重现E29提议动作＋事后真实门控的25.67%／30.97%，而不是E29实际执行的11.59%／24.62%。增加到Top8把理想覆盖提高到58.91%／61.72%，是实质性的候选信息，但仍遗漏约四成汇总Oracle收益。高正收益机会与低最佳动作命中说明“能找到一个有利动作”与“能找到高价值动作”是不同能力。','', '|seed|策略|L|恢复|逐例恢复中位数|平均绝对regret|P90绝对regret|触及页面均值|P90页数|最大页数|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for x in summary:lines.append(f"|{x['seed']}|{x['policy']}|{x['L']}|{x['recovery']:.2%}|{x['median_event_recovery']:.2%}|{x['mean_regret']:.6g}|{x['p90_regret']:.6g}|{x['unique_pages_mean']:.3f}|{x['unique_pages_p90']:.1f}|{x['unique_pages_max']}|")
 lines+=['','Set的Top8上限优于Independent Top8的40.07%／39.26%，说明集合评分在候选召回上仍有价值。但不同排序产生不同候选集合，这不是对验证后相同信息Set/Independent决策的部署比较。后者尚未运行。Top-L最多可触及2L页；本数据Top8实际平均约8.1页、最大9页，不能将Top8上限套用到“验证8页”的固定成本承诺。','', '## 建议B的真实候选空间上限','', '分别取Set与Independent预测前2个动作。首先统计最多4个动作的去重并集；然后用其触及的已选页和未选页组成所有合法笛卡尔交换，与原Medium取交集。后者才是建议B的受限面板。两种上限均用原Medium为分母，未把受限Oracle当作100%的Medium收益。','', '|seed|候选空间|Medium收益恢复|正收益机会|动作数均值|页数均值|最大页数|理想汇总NMSE改善|','|---|---|---:|---:|---:|---:|---:|---:|']
 for x in poolsummary:lines.append(f"|{x['seed']}|{x['pool']}|{x['recovery']:.2%}|{x['positive_coverage']:.2%}|{x['actions_mean']:.3f}|{x['unique_pages_mean']:.3f}|{x['unique_pages_max']}|{x['pooled_nmse_improvement_pct']:.3f}%|")
 lines+=['','理论上该方案最多8页、16个合法配对；本批数据实际最多6页，平均约4.9页。补充交叉配对只把恢复从约41%提高到约43%–44%。这43%–44%已经假设可以识别受限池内最佳真实动作并完美no-op；精确β验证仍保留有误差的预测响应与背景β，不能保证达到它。','', '受限池仍有一定剩余空间，不能由本实验断言任何验证策略都毫无价值；当前证据是不足以支撑“少数页验证即可恢复大部分Medium Oracle收益”。不据结果追加更大的L、重新设计候选池或执行B。','', '## 分层与遗漏收益','', '|seed|L|轨迹|恢复|正收益机会|','|---|---:|---|---:|---:|']
 for r in general:
  if r['dimension']=='trajectory':lines.append(f"|{r['seed']}|{r['L']}|{r['value']}|{float(r['recovery']):.2%}|{float(r['positive_coverage']):.2%}|")
 lines+=['','|seed|Top8未恢复收益合计|最大10个遗漏事件占比|','|---|---:|---:|']
 tail=[]
 for seed in [17,29]:
  rr=[r for r in rows if int(r['seed'])==seed and r['policy']=='Set' and int(r['L'])==8];rr=sorted(rr,key=lambda r:float(r['regret']),reverse=True);total=sum(float(r['regret']) for r in rr);share=sum(float(r['regret']) for r in rr[:10])/total;lines.append(f'|{seed}|{total:.6g}|{share:.2%}|');tail.extend(rr[:10])
 table('largest_top8_regrets.csv',tail)
 lines+=['','完整分层见generalization.csv；top8_action_distribution.csv保留每个提议动作的真实收益与排名，topL_per_event.csv保留候选内min/P10/median/max/mean/std与regret，避免只展示最佳动作。','', '## 假设验证成本，不是本轮实际开销','', '|验证页数|相对41页精算KV读取|相对21页精算KV读取|额外QK/层|','|---:|---:|---:|---:|']
 for n in [2,4,8]:lines.append(f'|{n}|{n/82:.2%}|{n/42:.2%}|{2*864*16*n*64*128/1e9:.3f} GFLOPs|')
 lines+=['','|seed|实际提议页池平均页数|假设稳态额外KV比例|混合历史长度平均额外KV比例|假设每层QK（按平均页数）|','|---|---:|---:|---:|---:|']
 for seed in [17,29]:
  x=ps[seed,'Touched-page-cartesian'];lines.append(f"|{seed}|{x['unique_pages_mean']:.3f}|{x['unique_pages_mean']/82:.2%}|{x['mean_extra_KV_fraction']:.2%}|{x['mean_hypothetical_QK_GFLOPs_layer']:.3f} GFLOPs|")
 lines+=['','只计算Key、M16、D128、12 heads×72 tiles。当前读出的候选页规模并非完整层各tile真实分布，此处仅作平均页数成本估计。不含既有Reader、集合评分、非连续Gather、同步和最终Attention，未假设与最终精算复用。实际新增Key验证次数为0，没有测得延迟或部署加速。','', '## 审计与停止','']
 for g in gate['results']:lines.append(f"- seed{g['seed']}：收益≥70% {'通过' if g['recovery_ge70'] else '未通过'}；正收益机会≥80% {'通过' if g['positive_coverage_ge80'] else '未通过'}。")
 lines+=['', f"独立堆排序重算576事件、{audit['topL_rows']}个Top-L记录、{audit['restricted_pool_rows']}个受限池记录；检查嵌套、单调、合法性、页数与源缓存SHA。12事件Independent长精度重算通过。Top1与已有事后真门控重放一致。没有原始QKV/Attention/Test访问。",'', 'E35-A回答了召回上限，尚未回答候选页真实β能否让现有响应模型实现这一上限。按参考门槛和本轮范围，E35-B未执行，也没有新增网络或Key验证模块。','', '相对目录：artifacts/e35_candidate_recall_v1/。主交付：topL_summary.csv、topL_per_event.csv、top8_action_distribution.csv、restricted_pool_summary.csv、restricted_pool_per_event.csv、generalization.csv、gate.json、independent_audit.json。s17/s29保存紧凑512动作重放缓存；精简包只含报告/CSV/源码/协议/审计，不含原始QKV或模型。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
 entries={p.name:p for p in DEST.glob('*.csv')}
 for n in ['REPORT.md','protocol.json','gate.json','summary.json','pool_summary.json','complete.json','sources.json','independent_audit.json']:entries[n]=DEST/n
 for p in (ROOT/'local_experiments/e35_candidate_recall').glob('*'):
  if p.is_file():entries['source/'+p.name]=p
 inventory={n:sha(p) for n,p in entries.items()};dump(DEST/'review_inventory.json',inventory);entries['review_inventory.json']=DEST/'review_inventory.json'
 with tarfile.open(DEST/'e35_a_review.tar','w') as tar:
  for n,p in entries.items():tar.add(p,arcname=n)
 with tarfile.open(DEST/'e35_a_review.tar') as tar:
  for n,h in inventory.items():assert hashlib.sha256(tar.extractfile(n).read()).hexdigest()==h
 dump(DEST/'package_audit.json',dict(files=len(entries),bytes=(DEST/'e35_a_review.tar').stat().st_size,sha256=sha(DEST/'e35_a_review.tar'),all_inventory_hashes_verified=True));print('FINISHED',len(entries),gate['passed'])
if __name__=='__main__':main()
