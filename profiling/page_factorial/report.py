import csv,json
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent
main=here.parent/'module_ablation'
events=json.loads((here/'events.json').read_text())
assert len(events)==192
def mean(rows,k):
    r=[x for x in rows if k in x]
    return np.average([x[k] for x in r],weights=[x['energy'] for x in r])
lines=['# Page64 外层分页 × R4 内组','',
'代码参照 7f63675893d369e233d732c6973cb6d5a454c488。原权重、条件配额、sparse clean、原七层三头 step0 换页范围。首 chunk 四格均使用部署分组及锚点重聚集，当前页维持自然顺序。历史 chunk 的学习特征均在 4608 token 上计算。GN 在已确定的页内按原 token ID 排序。','',
'| 四格 | 全层抽样 NMSE | 4 条 latent MSE | 4 条全片 LPIPS | 5 秒生成秒 | 20 条 LPIPS | 20 条 SSIM | 20 条 PSNR |','|---|---:|---:|---:|---:|---:|---:|---:|']
scores={}
for n in ['NN','NG','GN','GG']:
    a=np.load(main/f'sparse_{n}_local.npy').sum((0,1,2));r=json.loads((main/f'sparse_{n}.json').read_text())
    vals=[json.loads(f.read_text()) for f in sorted((main/'natural_validation'/n).glob('*.json'))]
    assert len(vals)==20 and len(r)==4
    scores[n]=np.array([x['lpips'] for x in vals])
    v=[a[0]/a[1],*[np.mean([x[k] for x in r]) for k in ['mse','lpips','seconds']],*[np.mean([x[k] for x in vals]) for k in ['lpips','ssim','psnr']]]
    lines.append('| '+n+' | '+' | '.join(f'{x:.6f}' for x in v)+' |')
lines+=['','4 条为 city_walk、short_721 × seed 721/722，80 帧。20 条为既有独立五场景 × seed 703–706，9 chunk，RGB [21,101)。视频随自身轨迹演化。局部机制对照固定输入、存活 token 以及预算，闭环保留原淘汰规则。','',
'| 四格 | 全摘要 | 固定同外页自然内序初选 mask | Risk | Reader | 池内最优一步 | 全分区最优一步 | 精算概率质量 | 精算 token 与 NN 的 Jaccard |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for n in scores:
    r=[x for x in events if x['cell']==n]
    lines.append('| '+n+' | '+' | '.join(f'{mean(r,k):.6f}' for k in ('summary','fixed','risk','reader','pool','legal','selected_mass','token_jaccard_vs_NN'))+' |')
lines+=['','48 个相同 Q64 事件：四条教师轨迹、chunk 2/4、层 3/8、头 3/4/9、step 0、每位置一个轮转 tile。每个交换对整块 Q64 共享；FP32 页分子分母、可选不换页，归一化 SSE 按 dense 输出能量加权。样本只覆盖启用纠偏的部分位置。','',
'| 四格 | 当前 −1 | 当前 +1 | 历史 −1 | 历史 +1 |','|---|---:|---:|---:|---:|']
for n in scores:
    r=[x for x in events if x['cell']==n]
    result=[]
    for k in ['c-1h0','c1h0','c0h-1','c0h1']:
        valid=[x for x in r if k in x]
        result.append(f'{mean(valid,k):.6f} / {mean(valid,"risk")-mean(valid,k):+.6f} ({len(valid)})')
    lines.append('| '+n+' | '+' | '.join(result)+' |')
lines+=['','配额点只重新运行 Risk 初选；每格为 NMSE / 配对误差下降（括号为可行事件数）。具体事件、分母能量和独立读出核对保存在 events.json。']
for n in ('NN','NG','GN'):
    lines.append(f"\n{n} 在 {(scores[n]<scores['GG']).sum()}/20 条上 LPIPS 优于 GG。")
lines+=['','## 配额重标定','',
'四条 7-chunk 标定轨迹为 forest_run、kitchen_cook 各用 seed 113/557。每格独立扫相同 6 个非锚点总预算 × 9 个当前份额。按窗口分别限制平均预算为非锚点 10%，同一步同层各头共享总长度。前四步和 clean 分开标定。独立局部留出为 snow_ski/917、bird_flight/919，每个 query-head 单元取一轮转行。闭环沿用四条 80 帧轨迹。','',
'| 分页 | 留出一 NMSE，旧表 → 新表 | 留出二 NMSE，旧表 → 新表 | 4 条 LPIPS，旧表 → 新表 | 4 条 latent MSE，旧表 → 新表 |','|---|---:|---:|---:|---:|']
repo=here.parents[1]
legacy=np.concatenate([np.load(f)['10_window_step_layer_head'] for f in (repo/'reproduction/vbench_current/policies.npz',here.parent/'clean_forward/policies.npz')],axis=1)[:5]
for n in ('GG','NG','NN'):
    cal=here/'calibration'/n
    fit=json.loads((cal/'fit.json').read_text())
    assert all(v<=target+1e-9 for _,_,v,target in fit['costs'])
    items=[]
    for scene,new in fit['holdout_nmse']:
        x=np.load(cal/(scene+'.npz'))
        before=np.take_along_axis(x['sse'][2:],legacy[...,None],axis=-1).sum()/x['dense_energy'][2:].sum()
        items.append(f'{before:.6f} → {new:.6f}')
    raw=[json.loads((main/f'sparse_{case}.json').read_text()) for case in [n,'refit_'+n]]
    for key in ('lpips','mse'):
        items.append(' → '.join(f'{np.mean([r[key] for r in rs]):.6f}' for rs in raw))
    lines.append('| '+n+' | '+' | '.join(items)+' |')
audit=[r['replay_nmse'] for r in events if 'replay_nmse' in r]
lines.append(f'\nGG 的 Q64 页分解与实际稀疏核输出之间最大 NMSE 为 {max(audit):.3g}。')
lines+=['','## 重标定后同一 20 条视频','',
        '校准只使用七个 chunk，9-chunk 复验继续复用满窗表。按视频配对，比较画质；计时仅引用四条短片的同卡数据。','',
        '| 四格 | 原表 LPIPS | 新表 LPIPS | 新表 SSIM | 新表 PSNR | 新表 LPIPS 优于各自原表 |','|---|---:|---:|---:|---:|']
with (here/'paired_video.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['case','prompt_seed','lpips','ssim','psnr'])
    for n in ['GG','NN','NG','GN','refit_GG','refit_NN','refit_NG']:
        files=sorted((main/'natural_validation'/n).glob('*.json'))
        assert len(files)==20
        vals=np.array([[json.loads(x.read_text())[k] for k in ('lpips','ssim','psnr')] for x in files])
        for ff,row in zip(files,vals):w.writerow([n,ff.stem,*row])
        if n.startswith('refit_'):
            old=scores[n[6:]]
            lp,ss,ps=vals.mean(0)
            lines.append(f'| {n[6:]} | {old.mean():.6f} | {lp:.6f} | {ss:.6f} | {ps:.4f} | {(vals[:,0]<old).sum()}/20 |')
(here/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
with (here/'events.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in events for k in r)));w.writeheader();w.writerows(events)
