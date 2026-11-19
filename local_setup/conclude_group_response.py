"""Conclusions tied to the completed predeclared frozen protocol."""
import json,csv,statistics,hashlib,shutil,datetime
from pathlib import Path
src=Path(__file__).resolve().parents[1];out=src/'artifacts/group_response_178f95a8';raw=Path('/tmp/group_response_results')
audit=json.loads((out/'final_audit.json').read_text());assert audit['passed']
local=json.loads((out/'local_aggregate.json').read_text());ix={(r['group'],r['panel']):r for r in local}
vs=list(csv.DictReader((out/'video_aggregate.csv').open()));v={(r['group'],r['level']):r for r in vs}
cost=list(csv.DictReader((out/'group_commit_cost.csv').open()));ci={(r['group'],r['component']):float(r['mean_ms']) for r in cost}
paired=[]
for c in ['horse_field','traffic_crossing']:
 for seed in [101,303]:
  for level in ['L2','L3']:
   root=raw/'video'/f'{c}_s{seed}';a=json.loads((root/'G-KVR'/level/'result.json').read_text());b=json.loads((root/'G-K'/level/'result.json').read_text())
   paired.append(dict(case=c,seed=seed,level=level,lpips_delta=a['quality']['lpips']-b['quality']['lpips'],latent_nmse_delta=a['latent_nmse']-b['latent_nmse']))
(out/'joint_paired_video.json').write_text(json.dumps(paired,indent=2))
lines=['实验结论（仅适用于本次固定权重、固定gate、固定预算与小样本）','', '1. K-response优于当前V-response单支。表示面板NMSE为0.314537 vs 0.373575；risk混合NMSE为0.046610 vs 0.048826。L2/L3的4case平均latent NMSE、LPIPS、SSIM和PSNR均支持K单支。V-response仍依赖K生成attention权重，不能解释为纯V与纯K比较。','2. 联合组没有稳定、明显的局部优势。表示NMSE仅比K低约0.018%；risk能量加权NMSE降低约0.60%，但仅47.5%事件胜过K。mass能量加权NMSE降低约0.51%，等权事件均值反而略差；不能只挑一种聚合方式宣称普遍改善。','3. 联合的闭环收益取决于稀疏层数。L2平均LPIPS由0.117140降至0.108830（约7.09%），latent NMSE由0.148176降至0.129522。L3平均LPIPS仅由0.334758降至0.333931（约0.25%），latent NMSE由0.445227降至0.441603；不足以据此认定联合分组必要。','4. 当前联合的预处理增量较小：同一真实clean KV上的commit总成本为7.400 ms/层/chunk，K为7.046 ms，增量约0.354 ms（5.02%）。这些是原参考实现的独立成本测量，不是上一轮融合kernel或全视频部署速度。','5. G-ORDER不可忽略：摘要表示NMSE 0.348750，逊于K/联合；但risk混合NMSE 0.013179，明显优于特征分组；L3匹配Dense的平均LPIPS 0.274602也更低，commit约2.484 ms。说明更好的摘要表示本身不保证在当前10%精确预算与路由下有更好的总体表现。L2联合更好、L3原顺序更好，不能宣称一个分组在所有层级都占优。','6. 本实验不支持“联合必不可少”。若保留特征分组，K是更强的单支参照；联合可作为L2的有益组合，但L3收益很小。应保留G-ORDER作为后续主对照。本次只做当前risk闭环，未运行mass闭环，不能外推联合在mass闭环下的收益。','', '完整性与边界：','120事件、4×2局部矩阵、独立表示诊断、16个匹配Dense+32个L2/L3视频全部完成。真实局部身份检查480组，闭环提交身份检查12,528次；无token缺失/重复，最老部分chunk保持原页/原token。','chunk2重分144/144页；chunk5/8重分288/338页，50页不动。稳态重分85.2071%，全部局部事件按历史页加权87.8049%，事件平均90.1381%。','同状态Dense排列一致性是数值一致，不是逐位相同：最大NMSE 5.4572e-11。原G-KVR特征/分组对指定提交另有逐位校验。','gate均值0.009966885，范围[0.009838214,0.010071835]；未调整。所有骨干和稀疏参数冻结，pool_feature_fn始终为原定义。','G-VR存在极端分母误差，未删除或截断：traffic_crossing/seed303/chunk5/layer29/step2表示面板相对分母误差约8.787e51。普通FP32统计会溢出，失败事件改用FP64统计后已补跑；log分母误差和原始相对误差均保留。','每组闭环使用匹配缓存协议的Dense参照，参照也会因浮点累积和淘汰集合发生差异。因此LPIPS/SSIM/PSNR衡量对各自匹配Dense的接近程度，不等同于统一目标的绝对视觉排名。两个prompt×两个seed不能支持统计显著性或跨场景普适性结论。','', '适配结果：未运行（条件未触发），不是已完成适配。预先记录的诊断是uniform历史pool比冻结pool的表示NMSE低>5%且至少60%事件获益。最强单支K及联合的uniform/learned误差比为1.00968/1.00992，均只有22.5%事件获益，未出现该pool不适配信号；这不能证明任何摘要适配都无收益。指定1,392次更新等完整条件保存在adaptation_protocol.json。','', '交付入口：REPORT.zh.txt完整表；local_matrix_4x2.csv；representation_panel.csv；video_per_case.csv；group_commit_cost.csv；page_variance.csv；gate_distribution.json；token_identity_checks.json；final_audit.json；四个case的.contact.png与video/<case>/<group>/<Dense|L2|L3>/video.mp4。']
(out/'CONCLUSIONS.zh.txt').write_text('\n'.join(lines)+'\n')
p=out/'adaptation_protocol.json';x=json.loads(p.read_text());x['status']='not_run_condition_not_triggered';p.write_text(json.dumps(x,indent=2))
p=out/'protocol.json';x=json.loads(p.read_text());x.update(status='complete_frozen_experiment',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
paths=list((src/'local_experiments/group_response').glob('*.py'))+list((src/'local_setup').glob('*group_response.py'))
(out/'code_manifest.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
for p in (src/'local_experiments/group_response').glob('*.py'):shutil.copy2(p,Path('/tmp/group_response_178f95a8/local_experiments/group_response')/p.name)
print('\n'.join(lines[:9]))
