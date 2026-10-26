from pathlib import Path
import json
root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt11_finetune_20261025'
def read(f):return json.loads(f.read_text())if f.exists()else{}
print('Opt11：部署数值对齐微调（Compact / 激进 Skip0 + FP16 PV）')
print('首 chunk / layer0 Dense；其它29层独立路由，约15%保留；冻结 backbone / prototype。\n')
c=[read(root/f'capture/rank{i}.json')for i in range(8)];print(f"补采分组 Q：{sum(x.get('completed',0)for x in c)}/288 事件")
print('训练前一致性检查：'+('通过'if read(root/'preflight.json').get('passed')else'进行中'))
completed=list((root/'training').glob('layer*/completed.json'));print(f'双分支训练：{len(completed)}/29 层完成')
labels={'preparing':'准备','labeling':'重算实际 AV 标签','training':'微调','validating':'实际 AV 验证','completed':'完成','loading':'加载','evaluating':'视频评估'}
for i in range(8):
 d=read(root/f'training/rank{i}.json')
 if d.get('status')!='completed':print(f"  GPU{i}: {labels.get(d.get('status'),'等待')}  layer={d.get('layer','-')}  {d.get('updates',d.get('completed',0))}/{d.get('total',200)}")
for v in ['Compact','Aggressive']:
 initial=[];best=[]
 for f in completed:
  folder=f.parent/v;a=read(folder/'initial_validation.json');b=read(folder/'best_validation.json');initial.append(a['nmse']);best.append(b['nmse'])
 if initial:print(f"  {v}: 已完成层平均实际 NMSE {sum(initial)/len(initial):.6f} → {sum(best)/len(best):.6f}")
strong=list((root/'strong_training').glob('layer*/completed.json'));print(f'第二阶段（学习率 1e-3）：{len(strong)}/29 层完成')
for i in range(8):
 d=read(root/f'strong_training/rank{i}.json')
 if d and d.get('status')!='completed':print(f"  GPU{i}: {labels.get(d.get('status'),'准备')} layer={d.get('layer','-')} {d.get('updates',d.get('completed',0))}/{d.get('total',200)}")
if len(strong)==29:
 for v in ['Compact','Aggressive']:
  a=[read(root/f'training/layer{i:02d}/{v}/initial_validation.json')['nmse']for i in range(1,30)];b=[read(root/f'strong_training/layer{i:02d}/{v}/best_validation.json')['nmse']for i in range(1,30)]
  print(f"  最终 {v}: {sum(a)/29:.6f} → {sum(b)/29:.6f}，下降 {(1-sum(b)/sum(a))*100:.3f}%")
q=[read(root/f'quality/rank{i}.json')for i in range(8)];print(f"完整视频评估：{sum(r.get('status')=='completed'for r in q)}/8 样例")
for i,r in enumerate(q):
 if r.get('status')and r.get('status')!='completed':print(f"  视频{i}: {labels.get(r.get('status'),r.get('status'))} {r.get('variant','')}")
pairs=[read(root/f'paired/prompt{i}/status.json')for i in range(8)];print(f"同卡配对测速：{sum(r.get('status')=='completed'for r in pairs)}/8 样例")
for phase in ['capture','training','strong_training','quality','paired']:
 codes=read(root/phase/'exit_codes.json')
 if codes and any(codes):print('失败：'+phase+'，退出码 '+str(codes)+'；详见该目录 rank*.log')
print('\n实验目录：'+str(root))

if (root/'REPORT.md').exists()and(root/'final_verification.json').exists():
 print('\n全部完成。报告：'+str(root/'REPORT.md'))
 summary=read(root/'final_summary.json')
 for name,t in summary['timing'].items():print(f"  {name}: 生成 {t['generation_seconds']:.3f} s")
