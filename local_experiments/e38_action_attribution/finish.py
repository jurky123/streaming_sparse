from .core import *
from .report import table
import tarfile,hashlib

def loadcsv(name):return list(csv.DictReader((DEST/name).open()))
def finish():
 audit=json.loads((DEST/'audit.json').read_text());gates=json.loads((DEST/'conditional_gates.json').read_text());assert not gates['C_pass'] and not gates['D_pass'];summary=loadcsv('role_weight_ablation.csv');bm=loadcsv('action_margin_summary.csv');reg=loadcsv('region_summary.csv');rescue=loadcsv('endpoint_rescue_summary.csv');ladder=loadcsv('teacher_ladder_summary.csv');cond=loadcsv('conditional_summary.csv');weights=loadcsv('weight_layer_shares.csv');front=loadcsv('acceptance_frontier.csv')
 lines=['# E38 动作翻转归因与角色感知信息诊断','','**已完成B和A；C极值诊断、D训练集阈值校准未启动。** D仅完成原E29固定提议的可分性诊断。两个模型种子各288个开发事件，来自两条轨迹。没有神经网络训练、摘要更新、新QKV采集或Test访问。','', '因E30缓存没有完整真实局部响应与128维摘要输出，用户明确授权离线读取已有开发KV重建冻结C摘要。仅B重建使用这些数据，A直接使用缓存；不是新增在线探测，也没有改变部署预算。','', '## 主要结果','','1. **质量和响应错误共同影响动作间隔。** 两种子β的平均Shapley式贡献均大于响应，但联合干预明显非加性；背景区域的有符号贡献也可能为负。不能将β单独宣布为唯一瓶颈。','2. **修复原动作不等于修复决策。** 同时提供原错误动作两个端点的真实r/β后，一半以上原动作被降至非正，但重新扫描后仍频繁选择另一个有害动作。','3. **角色区分有教师价值，敏感度权重没有稳定改善。** G4-mass与G4-direction在两个种子均优于对应G2，但仍有大量退化；W_J没有产生更可靠的校准。','4. **没有启动条件分支。** 端点仅β替换拒绝比例低于锁定的25%门槛；接受特征在seed29 snow轨迹未达到AUC0.60。没有扩统计、换阈值或只选择seed17继续。','', '## 评价单位和信息边界','','原E29、Risk mask、own保护、精算预算、M16、Medium512和一次交换/no-op均固定。初始面板512动作，五状态只用共同合法子面板。动作真值来自原Q64缓存。所有新间隔与预测分数统一除以Q64完整Dense平均能量+1e−12/64；这是离线正常数，不影响排序或no-op。旧M16缓存使用不同正归一化，按动作重放核验，而非把原始分数直接混减。','', 'j_pred和j*都包含no-op；非正最大收益选择no-op，动作索引打破平局。相同动作保留，间隔为0。贡献与收益均保留负值。低于1e−12的真收益变化用于近零计数，执行规则仍以预测>0判断。','', '## B2 响应与质量四顶点','','|seed|总间隔错误 ΣΔflip|响应贡献 ΣΦr|质量贡献 ΣΦβ|非加性交互 ΣΓ|剩余 M11−Mtrue|相同动作事件|','|---|---:|---:|---:|---:|---:|---:|']
 for s in [17,29]:
  x=next(x for x in bm if int(x['seed'])==s and x['dimension']=='ALL' and x['subset']=='all');lines.append(f"|{s}|{float(x['flip_error_sum']):.6f}|{float(x['phi_response_sum']):.6f}|{float(x['phi_beta_sum']):.6f}|{float(x['interaction_sum']):.6f}|{float(x['remaining_teacher_error_sum']):.6f}|{x['same_action']}|")
 lines+=['','Φr+Φβ=M00−M11，另加M11−Mtrue才等于完整间隔错误。M11是真实B16/M16，不是完整Q64目标；剩余项含Query和投影差异，不能强行分摊给两预测分支。每事件、layer、轨迹的均值/中位数/符号见action_margin_summary.csv。负Γ表示该间隔下联合替换相对单独替换存在额外修复，但不意味着训练因果关系已确定。','', '## B3 三页面区域','','|seed|端点贡献和|其他候选贡献和|背景贡献和|平均端点页数|','|---|---:|---:|---:|---:|']
 for s in [17,29]:
  x=next(x for x in reg if int(x['seed'])==s and x['dimension']=='ALL');lines.append(f"|{s}|{float(x['phi_endpoint_sum']):.6f}|{float(x['phi_other_sum']):.6f}|{float(x['phi_background_sum']):.6f}|{float(x['mean_endpoint_pages']):.3f}|")
 lines+=['','三个区域互斥完备，每区域的r和β一起替换，完整8顶点并对6种顺序平均。端点含原动作和真实最佳动作最多4页，使用了Oracle动作身份，不能在线识别。区域贡献与B2分支贡献是两张不同分解表，不能相加。负背景贡献表示有限干预改变了抵消，而不是背景信息无用；总和和中位数差异较大，分层与原始8顶点都保留。','', '## B4 原错误动作端点修复','','|seed|原退化事件|仅β拒绝原动作|仅r拒绝原动作|r/β一起拒绝原动作|重新扫描正收益|重新扫描有害|重新扫描近零|no-op|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for s in [17,29]:
  z={x['mode']:x for x in rescue if int(x['seed'])==s and x['dimension']=='ALL'};x=z['both'];lines.append(f"|{s}|{x['n']}|{z['beta']['original_rejected']}|{z['response']['original_rejected']}|{x['original_rejected']}|{x['positive']}|{x['harmful']}|{x['near_zero']}|{x['noop']}|")
 lines+=['','只有双端点联合替换重新扫描512动作；其他两组仅检查原动作。修正后没有no-op，且仍有56/95与59/91事件选择有害动作。原来两个页面信息更准并没有补齐所有备选动作和背景的信息。不能据“拒绝原动作”推导接受安全性。','', '## B5 TrueBeta剩余错误','','下表固定TrueBeta原错误动作和真实最佳动作，不重新选动作。','', '|seed|信息|事件数|原错误动作变非正|固定偏好正确|','|---|---|---:|---:|---:|']
 for x in ladder:
  if x['dimension']=='ALL' and x['stage']!='true_Q64_perp':lines.append(f"|{x['seed']}|{x['stage']}|{x['n']}|{x['wrong_action_nonpositive']}|{x['fixed_preference_correct']}|")
 lines+=['','给真实响应后，M16/B16仍不能完全纠正这些事件；Q64/B16的固定偏好已接近完整Q64，但仍有若干原错误动作被判为正。完整Q64/128闭合原真实目标。Q64正交分解使用冻结B的同一列空间经FP64 QR正交化，保证g_full=g_parallel+g_perp；不重新拟合空间。M16真实响应仍使用原冻结B坐标，数值差单独核验。这些固定动作结果不是重新决策收益上界。','', '## A 六组角色与权重','','W_m=m²；W_v=m²(||r+c||²+1e−12)；W_J=||dE/dbeta||²+1e−12。权重全部在原预测点计算，不在教师替换后迭代。每个参考状态拟合一次并冻结评价动作；G2候选身份固定，G4按参考状态是否精算分角色。初始G1/G2正好是R16/A32。','', '|seed|方法|Set恢复|Independent恢复|逐例平均改善|退化/288|P10|最差绝对NMSE增加|','|---|---|---:|---:|---:|---:|---:|---:|']
 for s in [17,29]:
  for method in METHODS:
   x=next(x for x in summary if int(x['seed'])==s and x['method']==method and x['policy']=='Set');ii=next(x for x in summary if int(x['seed'])==s and x['method']==method and x['policy']=='Independent');lines.append(f"|{s}|{method}|{float(x['R_executed']):.2%}|{float(ii['R_executed']):.2%}|{float(x['mean_improvement_pct']):.3f}%|{x['degraded']}|{float(x['p10_pct']):.3f}%|{float(x['worst_absolute_nmse_increase']):.6g}|")
 lines+=['','四角色在mass和direction权重下提供了可观察的教师增益，但G4-direction仍退化72/288（两个种子均25%），seed29最差绝对增量明显高于E29。因此不是可部署成功，也不自动启动四偏移学习。W_J是局部预测残差敏感度，不是动作间隔敏感度；其失败不能证明所有敏感度监督都无效。ε在运行前固定，没有根据结果调节。','', '|seed|权重|layer29原始权重占比|layer29平均ε占比|','|---|---|---:|---:|']
 for x in weights:
  if int(x['layer'])==29:lines.append(f"|{x['seed']}|{x['weight']}|{float(x['raw_weight_fraction']):.2%}|{float(x['mean_epsilon_fraction']):.2%}|")
 lines+=['','这不是E37的加权误差能量比例：这里只累计本轮拟合权重。每组最大归一化权重、有效页数、偏移量及层级结果见weight_summary.csv；不能把W_J表现变化全部归因于导数结构而忽略尺度/ε。','', '## 五状态条件诊断','','|seed|方法|双重中心化相关|事件条件相关中位数|符号翻转召回|','|---|---|---:|---:|---:|']
 for x in cond:lines.append(f"|{x['seed']}|{x['method']}|{float(x['double_centered']):.4f}|{float(x['median_event_corr']):.4f}|{float(x['sign_flip_recall']):.2%}|")
 lines+=['','A的角色/权重在各参考状态重新计算，因此该条件指标评价完整的“状态角色+教师校准+集合评分”，不等同于冻结一套页面预测后的纯集合交互能力。绝不在同一参考状态的不同候选动作间重新拟合。','', '## 条件阶段与停止决定','','C启动规则在跑B前已写入protocol：两个种子β平均贡献均为正且大于响应，并且仅端点β替换拒绝原错误动作比例均≥25%。前一项通过，后一项仅21.05%/23.08%，所以未进一步读取原始logits做极值诊断，未计算半径、未试饱和函数。25%是本轮资源筛选规则，不是极值路线不可能有效的证明。','', 'D只完成提议可分性。规则要求两个种子、各轨迹至少一个特征AUC≥0.60，并且其预测收益三分位分箱AUC中位数≥0.60；有效箱每类至少5例。裕量I/(B+ε)在两个种子的条件AUC均约0.631，但seed29 snow的AUC为0.589；正probe比例也未补足。所以不使用训练轨迹校准网格，不产生冻结gate，不把全no-op当成功。','', '|seed|轨迹|裕量AUC|probe一致性AUC|','|---|---|---:|---:|']
 for s in [17,29]:
  for traj in ['kitchen_cook_s701','snow_ski_s702']:
   vals={x['feature']:float(x['auc']) for x in front if int(x['seed'])==s and x['scope']=='trajectory' and x['value']==traj};lines.append(f"|{s}|{traj}|{vals['rho']:.4f}|{vals['pi_positive']:.4f}|")
 lines+=['','此可分性筛选使用开发标签，是探索性可行性诊断，不是独立验证。真实初始NMSE未进入接受特征。I/B是归一化输出变化，不是E17的分子残差I/B/D；Cauchy必要条件B<4A只约束预测，不保证真实质量。','', '## 审计与成本','','576事件的分支/区域归因闭合、面板不变与count/own通过；E29、TrueBeta及G2-mass动作重放E30/E37。抽取跨层/轨迹事件，以NumPy longdouble直接正混合输出差核验4顶点、8区域和六组选中动作；dE/dbeta与自动微分独立比对。Q64投影与正交补使用同一完整能量分母，B5全部事件闭合。详见audit.json与源哈希。','', '重建的真实响应是离线标签，不能算在线额外信息。六组偏移也依赖真实β，未构造在线校准器。W_J若未来在线计算，需全页向量归约，不能按“几个标量”忽略成本；本轮不声称加速。没有新的部署缓存、Kernel或视频实验。','', '## 后续设计判断','','本轮支持优先关注动作端点与精算/近似角色，但同时表明：只修复当前动作端点会暴露其他错误候选，响应与质量不可简单拆开独立优化。角色教师的收益可以作为下一次表示设计的约束，不能直接变成新增四偏移模块。','', '更合适的下一步是判断当前在线摘要能否识别端点间的响应/质量对比及其误差，而不是继续优化全页β MSE。若需要新表示，应保持初选和真实混合输出质量，用同信息Set/Independent/no-op验证；当前结果没有授权扩大Reader、增加原始Key探测或直接实现接受gate。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n');dump(DEST/'stage_status.json',dict(B='complete',A='complete',C='not_started_gate_failed',D='proposal_diagnostic_complete_threshold_calibration_not_started',test='sealed',training=False))
 (ROOT/'local_experiments/e38_action_attribution/README.md').write_text('''# E38 action attribution

Run B for seeds17/29, then report --stage B; run A for seeds17/29, report --stage A; audit; finish. Modules under local_experiments.e38_action_attribution. CUDA FP64 and canonical E24/E30 caches required. B reconstructs frozen summary and true local response from existing dev KV with explicit user authorization; A uses cache. No new capture or Test.

Protocol is fixed before runs. C and D calibration gates failed; no extreme statistics or threshold training performed. Finished runs refuse overwrite. Pilot --limit does not mark completion. Report archive contains tables/source/audit, not full QKV or large per-event targets. All gains in attribution use a common Q64 full-energy normalizer; M16 selection replay is unchanged.
''')
 src=loadcsv('source_hashes.csv');hm={x['path']:x['sha256'] for x in src}
 for f in (ROOT/'local_experiments/e38_action_attribution').iterdir():
  if f.suffix in ['.py','.md']:hm[str(f.relative_to(ROOT))]=sha(f)
 hm[str((DEST/'protocol.json').relative_to(ROOT))]=sha(DEST/'protocol.json');table('source_hashes.csv',[dict(path=k,sha256=v) for k,v in sorted(hm.items())]);items=sorted([f for f in DEST.iterdir() if f.suffix in ['.md','.json','.csv'] and f.name!='package_manifest.json']+list((ROOT/'local_experiments/e38_action_attribution').glob('*.py'))+[ROOT/'local_experiments/e38_action_attribution/README.md']);manifest=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in items];dump(DEST/'package_manifest.json',manifest)
 with tarfile.open(DEST/'e38_review.tar','w:gz') as tar:
  for f in items+[DEST/'package_manifest.json']:tar.add(f,arcname=str(f.relative_to(ROOT)))
 with tarfile.open(DEST/'e38_review.tar','r:gz') as tar:
  for r in manifest:assert hashlib.sha256(tar.extractfile(r['path']).read()).hexdigest()==r['sha256']
 print('PACKAGE VERIFIED',len(items),(DEST/'e38_review.tar').stat().st_size,flush=True)
if __name__=='__main__':finish()
