"""Package verified E41 evidence and update research reading path."""
from .run import *
import shutil,tarfile

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run():
 audit=json.loads((D/'audit.json').read_text());assert audit['passed'] and audit['partition_events']==2880 and audit['independent_events']>=20
 for seed in [17,29]:
  done=json.loads((D/f's{seed}/complete.json').read_text());assert done['events']==1440 and done['parameter_events']==48
  old={}
  for split in ['train','development']:
   for r in json.loads((CACHE/f'{split}_s{seed}/sources.json').read_text()):old[r['event']]=r
   assert done['model_sha']==json.loads((CACHE/f'{split}_s{seed}/capture_complete.json').read_text())['reader_sha']
  for r in json.loads((D/f's{seed}/sources.json').read_text()):
   assert r['event_sha']==old[r['event']]['event_sha'] and r['summary_sha']==old[r['event']]['summary_sha']
 audit.update(all_input_hashes_match_E40=True,reader_checkpoint_matches_E40=True);dump(D/'audit.json',audit)
 rows=read(D/'decision_summary.csv');heads=[r for r in rows if r['dimension']=='overall'];txt=[]
 def put(s):txt.append(s)
 put('# E41：训练信号覆盖与梯度分配审计\n')
 put('**本轮完成，无训练、无模型更新、无原始KV读取，Test继续封存。完整训练集也未学到可靠集合决策；发现了局部响应监督对部分真实重要、摘要低估页面的覆盖缺口，但没有证明β梯度普遍消失或共享层长期抵消是主要原因。**\n')
 put('## 范围与证据口径\n')
 put('每种子1152个训练事件（8条轨迹）及288个开发事件（2条轨迹），合计2880个seed-event。事件在同一轨迹内相关，不是2880条独立视频。固定E29最终选定Reader、C摘要、Adaptive φ、B16、Risk、Medium和M16；实际收益按Q64完整输出评价。训练集评价的是最终冻结模型，不是历史训练中的在线表现。\n')
 put('参数梯度为每种子48个预先固定的训练Risk状态，覆盖8条轨迹×3层×2条件。排序四分位选样恰好集中在head6，因此不能把该参数子样本推广到其他head；完整输出梯度分析覆盖全部head。没有重放历史Adam状态或60/20/20全部训练状态，也不能声称观察到了长期梯度冲突。\n')
 put('## A：完整训练集能力\n')
 put('|seed|划分|策略|Oracle恢复|提议＋真门控恢复|H_gain|H_loss|退化|P10相对改善|\n|---|---|---|---:|---:|---:|---:|---:|---:|')
 for r in heads:put('|%s|%s|%s|%.2f%%|%.2f%%|%.6f|%.6f|%s/%s|%.2f%%|'%(r['seed'],r['split'],r['policy'],float(r['recovery_pct']),float(r['hindsight_recovery_pct']),float(r['H_gain']),float(r['H_loss']),r['harm'],r['events'],float(r['p10_relative_pct'])))
 put('\n训练Set恢复仅4.01%/0.64%，低于开发11.59%/24.62%。不能主要归因于跨轨迹泛化；也不能据此证明信息不可预测。训练seed29的Set不及同信息Independent，说明组合优势在训练分布上也不稳定。两个种子均有7/8条训练轨迹净正收益，但一条轨迹的大损害抵消大量收益，不能只报正轨迹数量。\n')
 put('|训练轨迹|seed17 Set恢复|seed29 Set恢复|\n|---|---:|---:|')
 ts={}
 for r in rows:
  if r['dimension']=='trajectory' and r['split']=='train' and r['policy']=='Set':ts.setdefault(r['value'],{})[r['seed']]=float(r['recovery_pct'])
 for n,v in sorted(ts.items()):put('|%s|%.2f%%|%.2f%%|'%(n,v['17'],v['29']))
 put('\n相对改善保留原`initial>1e-12`口径，没有截尾。训练逐例平均相对改善为−4869.07%/−976381.05%，由极小初始误差事件放大；应同时看绝对损害。最差事件分别为seed17 `forest_run/c8/l29/step0/head9/t39`：初始NMSE 2.256e−5、增加0.668344；seed29 `reef_fish/c5/l29/step0/head9/t37`：初始9.609e−8、增加1.010389。额外直接重算这些动作的混合目标核验，未删除异常事件。seed29最差动作的直接增量为1.010382，原减法增量为1.010389，相差6.47e−6；属于移出大质量页时的数值抵消，不改变有害性或报告百分点结论。完整指标及分层见`decision_summary.csv`。\n')
 put('## B：重要低估页面的监督覆盖\n')
 put('原始局部损失含m²；β为按页RMS归一化后的Huber，保留全部mean reduction。原β floor为1.5330186546，收益floor为0.0007554343，局部尺度为0.0064952183/0.0064888371。以下份额均先在事件内归一化，再对非空事件等权平均；不按layer29的大误差池化。\n')
 put('“低估”预定为m<α*/10。缓存配对表使用E40的Q64预测切片至M16，微小计算路径差异单独记录；主输出梯度表使用原E29 M16前向。\n')
 put('|训练子集|seed|非空事件|真实质量份额|局部响应损失份额|β梯度L1份额|KL参考梯度L1份额|\n|---|---:|---:|---:|---:|---:|---:|')
 cr=read(D/'mass_coverage_summary.csv')
 for role in ['all','harmful_predicted_endpoint','oracle_endpoint']:
  for r in cr:
   if r['split']=='train' and r['status']=='summary_under10' and r['role']==role:
    put('|%s|%s|%s|%.4f%%|%.4f%%|%.4f%%|%.4f%%|'%(role,r['seed'],r['events'],100*float(r['true_mass_share_mean']),100*float(r['local_loss_share_mean']),100*float(r['beta_gradient_share_mean']),100*float(r['kl_gradient_share_mean'])))
 put('\n两种子各968个训练事件包含摘要低估十倍的Query–Page对。在这些事件内，该子集约占9.46%/9.13%的Query–Page样本、10.11%/9.95%的真实质量，却只有0.0703%/0.0692%的局部损失份额，局部输出梯度L2份额约0.0431%/0.0416%。这是实际监督覆盖差异，不只是推理敏感性假设。它不能证明改成真实质量加权就能改善决策：更高权重可能破坏现有优化平衡。\n')
 put('β监督必须区别看待：全体低估子集仍获得约10.53%/10.06%的β梯度L1份额，接近其真实质量份额，**不支持“β梯度在所有低估页面上消失”**。但有害动作端点中的低估子集，在90/398、103/395个有害事件中出现：其真实质量份额约1.447%/1.213%，β梯度份额只有0.125%/0.122%，KL参考分别0.940%/0.757%。该端点子集的归一化β梯度总量相对不归一化Huber的中位比约0.104/0.097；这是公式参照，不是新训练结果。覆盖缺口只涉及部分错误，不能解释所有退化。\n')
 put('原β监督输出梯度在非零误差处始终指向真实β，未发现符号反转。原集合损失对有害端点β的梯度L1份额约4.52%/4.14%，并没有整体忽略这些端点；其方向与“朝真实β修正”一致的Query–Page比例约48.5%/49.0%。这不说明集合目标梯度错误：在预测响应不准、目标含Q64真值的情况下，它可能通过联合调整而非逐点物理回归纠错。不能把方向不一致直接判为bug。\n')
 put('KL参考梯度为(α̂−α*)/16，只诊断相对质量分布覆盖，不约束共同β尺度；没有用于任何更新。`mass_coverage_summary.csv`的条件均值不可与全事件均值直接相除。主角色表同时给出全事件、layer、head和轨迹分层，重叠角色不与互斥区域份额相加。\n')
 put('## C：共享参数梯度与裁剪\n')
 put('|冻结训练子样本|seed17|seed29|\n|---|---:|---:|\n|隐藏层local范数中位数|0.03273|0.02693|\n|隐藏层β范数中位数|0.04513|0.04658|\n|隐藏层set范数中位数|0.02916|0.01813|\n|隐藏层β/set余弦中位数|0.07487|0.11575|\n|隐藏层β/set负夹角比例|43.75%|35.42%|\n|隐藏层合梯度/各范数之和中位数|0.6896|0.6825|\n|全局裁剪触发比例|29.17%|31.25%|\n')
 put('β项在共享层的中位梯度范数并不小于另外两项；存在事件级冲突，但中位夹角并未显示β持续被反向压制。响应输出行的β直接梯度为0，β输出行的local直接梯度为0，符合结构；共享隐藏层才承接耦合。全局裁剪只乘同一标量，事件内分量比例和夹角不变。参数证据仅限head6、Risk状态、最终检查点，不能据此排除历史训练冲突，也不能声称发现“长期抵消”。\n')
 put('## 数值与来源审计\n')
 put('全2880事件的E40原动作及真实最佳动作身份一致；输入事件、摘要和Reader检查点哈希匹配E40。原局部/β解析梯度与自动微分一致。独立使用分别归一化的正质量混合公式核验集合分数及梯度，并核验KL梯度和真实动作收益；包含预定抽查及各划分最差动作。真实增量核验明确记录绝对差，极端动作采用rtol=1e−5、atol=1e−9；不把原增量公式称为逐位精确。详情见`audit.json`、`independent_checks.csv`。\n')
 put('原M16前向与E40 Q64缓存切片的最大β差约1.70e−4/2.11e−4，未改变任何动作；记录在`checks.csv`。份额守恒的最大FP32舍入误差约2.11e−7，容差1e−6。梯度绝对单位受mean reduction和尺度影响，因此报告按事件的份额、范数与方向，不能把不同量纲的绝对数直接相加。\n')
 put('## 当前判断与后续边界\n')
 put('1. **完整训练集也弱**：撤回“训练已解决，只剩泛化”的解释；尾部损害尤其集中，但没有删除它们。\n2. **发现有限的监督覆盖问题**：摘要低估页面的局部响应监督非常弱；部分错误端点的β监督也相对真实质量不足。β本身不是全局无梯度，集合梯度也在关注端点。\n3. **未证明普遍优化抵消**：冻结子样本不支持把失败归因于β被其他目标长期压制，裁剪本身不会改变事件内比例。\n4. 本轮仅支持把“质量低估与局部监督耦合”列为一个有证据的后续受控假设，而不是立即替换整个损失或加入KL。若后续获准做一次对照，应固定架构/预算，仅改变预先确定的一种覆盖机制，并同时检验训练真实收益、错误损害与开发收益；不进行多权重搜索。当前未启动该对照。\n')
 put('参数子样本不能覆盖实际最差的head9事件，故在选择涉及共享层优化的新方案前，应先核验该范围限制；本轮不据此追加训练。Test保持封存。\n')
 put('## 文件阅读顺序\n\n本报告 → `decision_summary.csv`（先训练轨迹，再层/head）→ `mass_coverage_summary.csv`及`gradient_role_summary.csv` → `parameter_gradient_summary.csv` → `audit.json`与`protocol.json`。事件级大表在同名artifact目录的`s17/`、`s29/`。代码见`local_experiments/e41_training_signal/`。\n')
 (D/'REPORT.md').write_text('\n'.join(txt))
 dump(D/'stage_status.json',dict(A='complete',B='complete',C='complete_limited_head6_Risk_snapshot',training='not_started',test='sealed',new_module=False,conclusion='Weak full-train decisions; local supervision undercoverage for summary-underestimated mass; no universal beta starvation or established long-term shared-gradient cancellation.'))
 evidence=ROOT/'docs/research/evidence/e41_training_signal_v1';evidence.mkdir(parents=True,exist_ok=True)
 small=['REPORT.md','protocol.json','stage_status.json','audit.json','independent_checks.csv','decision_summary.csv','gradient_role_summary.csv','parameter_gradient_summary.csv','mass_coverage_summary.csv','headline.json']
 manifestp=ROOT/'docs/research/evidence/manifest.json';manifest=json.loads(manifestp.read_text());manifest=[r for r in manifest if 'e41_training_signal_v1/' not in r['snapshot']]
 for name in small:
  shutil.copy2(D/name,evidence/name);manifest.append(dict(source=str((D/name).relative_to(ROOT)),snapshot=str((evidence/name).relative_to(ROOT)),sha256=sha(D/name)))
 dump(manifestp,manifest)
 package=[]
 for name in small:package.append(D/name)
 for seed in [17,29]:
  for name in ['complete.json','decisions.csv','output_gradient_roles.csv','parameter_gradients.csv','checks.csv','sources.json']:package.append(D/f's{seed}'/name)
 package.append(D/'mass_underestimation_coverage.csv')
 dump(D/'package_manifest.json',[dict(file=str(p.relative_to(D)),sha256=sha(p)) for p in package])
 with tarfile.open(D/'e41_review.tar.gz','w:gz') as tf:
  for p in package+[D/'package_manifest.json']:tf.add(p,arcname=str(p.relative_to(D)))
 print('PACKAGED',len(package),audit['independent_events'])
if __name__=='__main__':run()
