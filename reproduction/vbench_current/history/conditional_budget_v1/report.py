from pathlib import Path
import json,csv,gzip
import numpy as np
from calibrate import train,tests
D=Path(__file__).parent; prior=D.parent/'rebased_budget_ablation_v1'; weights=D.parent/'rebased_retrain_v1'
def read(p):return json.loads(p.read_text())
m=read(D/(train[0]+'.json')); probs=np.load(D/'policies.npz'); mods=read(prior/'module_rows.json')
lines=['# 分条件预算标定','','固定首帧 24 页。每个窗口用 6 档 total、每档 9 个名额分配比例得到离散可行的 p 和 q，另有当前全算加历史 5%/10% 两个对照。own 计入当前。quotas.csv 可查实际页数、p、q 和取整后的 total。','',
'四条标定轨迹为森林、厨房 × seed 113、557。每个 chunk、step、layer、head 采一个轮转查询行。各轨迹 NPZ 保留单元 × 配额的完整 SSE 与 Dense 能量；这里的误差用于同一输入的模块标定。闭环需要单独看成片。','',
'在同一平均计算预算上限下，用拉格朗日乘子选各单元离散预算。可执行方案每个层步调用内共享总页数，每个头可用不同的 p、q。每头可独立选择 total 的结果仅作参照。','',
'| 平均预算 | 标定单元 | 新提示词 | 新 seed | 两者均新：滑雪 | 两者均新：飞鸟 |','|---|---|---:|---:|---:|---:|']
r=read(D/'holdouts.json')
for i in range(0,len(r),4):
 z=r[i:i+4];lines.append(f"| {z[0]['budget']}% | {z[0]['gran']} | "+' | '.join(f"{x['ratio']:.3f}" for x in z)+' |')
lines+=['','表中为相对同档固定配额的局部 SSE 比；1 表示持平。window 仅按窗口，full_age 把满窗分为四段，chunk 为逐绝对 chunk。window_step_layer_head 使用满窗共享表。','',
'## 计算量与耗时','','| 档位 | 30 秒稀疏融合主项 TFLOP | Dense 同形吞吐折算 ms | 同形隔离模块耗时加权 ms | 兑现比例 |','|---|---:|---:|---:|---:|']
for budget in (10,15):
 tab=probs[f'{budget}_window_step_layer_head']; work=ref=actual=0
 with gzip.open(D/f'active_{budget}.csv.gz','wt') as f:
  wr=csv.writer(f);wr.writerow(['chunk','window','step','layer','head','total_nominal','A','B','p','q'])
  for j,c in enumerate(range(2,41)):
   w=min(432,(c+1)*72);lookup=np.array(m['counts'][str(w)])[tab[j]]
   for step,l,h in np.ndindex(4,29,12):
    k=int(tab[j,step,l,h]);a,b=lookup[step,l,h];wr.writerow([c,w,step,l+1,h,m['grid'][k][0],a,b,a/72,b/(w-96)])
   for total,n in zip(*np.unique(tab[j,:,:,0]//9,return_counts=True)):
    row=next(x for x in mods if x['pages']==w and x['spec']==[m['grid'][int(total)*9][0],0.25])
    work+=n*(row['exact_flops']+row['comp_flops']);ref+=n*row['at_dense_rate_ms'];actual+=n*row['fused_ms']
 lines.append(f'| {budget}% | {work/1e12:.2f} | {ref:.1f} | {actual:.1f} | {ref/actual:.1%} |')
times=[read(D/(x+'.json'))['wall_seconds'] for x in train+tests]
lines+=['',f'8 卡并行扫描，每轨迹 {min(times):.2f}–{max(times):.2f} 秒；离线选表与留出评估 {float((D/"calibration_seconds").read_text()):.2f} 秒。计算量含精算双项 AV 与摘要补偿主项；加权耗时引用已有同形融合模块测试，额外路由、查表开销由闭环 profile 单列。','']
for p in sorted(D.glob('[01]_*.json')):
 if 'repeat' in p.name:continue
 v=read(p)
 if 'wall' in v:lines.append(f"- {p.stem} 生成 {v['wall']:.3f} 秒，预热 {v['warmup']:.2f} 秒，路由纠偏 {v['stages']['route_and_correction']['milliseconds']:.1f} ms。")
(D/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
