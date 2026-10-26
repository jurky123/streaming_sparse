"""Full development frontier for paired heads; no video or speed claims."""
import argparse
import csv
import json
import os
from pathlib import Path
import torch
from .common import config,atomic_json,update,paired_seed,save
from .labels import GROUPS
from .learning import ActionHead,batch,evaluate


@torch.no_grad()
def margin(model,root,m,layer,kind,basis):
    train=[r for r in m['rows']if r['split']=='train'and r['history_tokens']]
    rows=[next(r for r in train if r['prompt_id']==pid)for pid in m['splits']['train']]
    errors=[[]for _ in range(12)]
    for row in rows:
        b=batch(root,m,[row],layer,m.get('monitor_beta',.5),kind,basis);pred=model(b['x'])
        for head in range(12):
            valid=b['valid'][head].clone();valid[:,0]=False
            errors[head].append((pred[head]-b['gain'][head]).abs()[valid].cpu())
    return torch.stack([torch.cat(values).quantile(.9)for values in errors])


def worker(root,m):
    torch.cuda.set_device(0);rank=int(os.environ.get('RANK',0))
    groups=m.get('groups',list(GROUPS));seeds=m.get('training_seeds',[0,1]);teams=m.get('train_layer_teams',1)
    size=len(groups)*len(seeds);assert int(os.environ.get('WORLD_SIZE',1))==size*teams
    team=rank//size;local=rank%size;seed=seeds[local//len(groups)];kind=groups[local%len(groups)];paired_seed(100+seed)
    val=[r for r in m['rows']if r['split']=='val'and r['history_tokens']]
    all_points=[]
    for layer in m['layers'][team::teams]:
        basis=torch.load(root/f'action_basis/layer{layer:02d}.pt',map_location='cpu',weights_only=True)
        normal=torch.load(root/f'normalization/layer{layer:02d}.pt',map_location='cpu',weights_only=True)[kind]
        for snapshot in ['last','best']:
            state=torch.load(root/f'P3_training/layer{layer:02d}/{kind}/seed{seed}/{snapshot}.pt',map_location='cpu',weights_only=True)
            model=ActionHead(normal).cuda();model.load_state_dict(state['head']);model.eval()
            margins=[('zero',0.)]
            if 'train_q90' in m.get('report_margins',['zero','train_q90']):
                tau=margin(model,root,m,layer,kind,basis).cuda()[:,None]
                save(root/f'P3_reports/calibration/{kind}.seed{seed}.layer{layer:02d}.{snapshot}.pt',dict(train_margin=tau.cpu(),kind=kind,seed=seed,layer=layer,updates=state['updates']))
                margins.append(('train_q90',tau))
            for beta in m['beta_grid']:
                for lam in m['lambda_grid']:
                    for margin_name,threshold in margins:
                        result=evaluate(model,root,m,val,layer,beta,kind,basis,lam,threshold)
                        all_points.append(dict(kind=kind,seed=seed,layer=layer,snapshot=snapshot,
                            updates=state['updates'],beta=beta,lam=lam,margin=margin_name,**result))
                        update(root/f'P3_reports/rank{rank}.json',status='evaluating',kind=kind,seed=seed,
                            layer=layer,snapshot=snapshot,beta=beta,lam=lam,margin=margin_name,completed_points=len(all_points))
    atomic_json(root/f'P3_reports/points_rank{rank}.json',dict(points=all_points))
    update(root/f'P3_reports/rank{rank}.json',status='completed',kind=kind,seed=seed,points=len(all_points))


def merge(root,m):
    workers=m.get('worker_count',8);layers=len(m['layers'])
    points=[p for rank in range(workers)for p in json.loads((root/f'P3_reports/points_rank{rank}.json').read_text())['points']]
    grouped={}
    for p in points:
        key=(p['kind'],p['seed'],p['snapshot'],p['beta'],p['lam'],p['margin'])
        grouped.setdefault(key,[]).append(p)
    rows=[]
    for key,values in grouped.items():
        assert len(values)==layers and len({p['layer']for p in values})==layers
        rows.append(dict(zip(['kind','seed','snapshot','beta','lambda','margin'],key),
            nmse=sum(p['nmse']for p in values)/layers,keep=sum(p['keep']for p in values)/layers,
            keep_H=sum(r['keep_H']for p in values for r in p['rows'])/(layers*96),
            keep_C=sum(r['keep_C']for p in values for r in p['rows'])/(layers*96),
            acceptance=sum(r['acceptance']for p in values for r in p['rows'])/(layers*96),
            accepted_worsening=sum(r['actual_worsening']for p in values for r in p['rows'])/(layers*96)))
    output=root/'reports';output.mkdir(exist_ok=True)
    with(output/'quality_cost_frontier.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(output/'summary.json',dict(status='completed_P3',scope=f'{layers}-layer local true128-d AV only; no network/closed-loop or speed result.',
        first_chunk_dense=True,first_chunk_anchor=False,history_prototypes={'nearest':8,'second':4,'older':2},
        current_prototypes=16,key64_status='newly trained candidate; full128 P1 reference retained',points=rows,
        train_seed_count=2,test_used=False,next_stage='P4 requires locking at most2 candidates and real-network validation; no automatic full29-layer efficacy claim.'))
    summary=json.loads((output/'summary.json').read_text())
    summary.update(training_objective=m.get('training_objective','gain_huber'),best_metric=m.get('best_metric','huber'),
                   target_keep=m.get('target_keep'),monitor_beta=m.get('monitor_beta',.5),
                   monitor_lambda=m.get('monitor_lambda',.1),report_margins=m.get('report_margins',['zero','train_q90']))
    atomic_json(output/'summary.json',summary)
    text=['# Slim-AV v0.10：距离P8/P4/P2、当前P16','',
        '首chunk生成Dense；后续其KV作为普通历史候选，不强制保留。骨干和LoRA冻结，新的分组/Key投影已重训，动作头为四组×两个Seed。','',
        f'当前为{layers}层离线真实AV实验。下表同beta/lambda仅作开发诊断，不是等成本结论；完整前沿见quality_cost_frontier.csv。',
        '', '| 组别 | Seed0 NMSE | Seed1 NMSE | Seed0总Keep | Seed1总Keep |','|---|---:|---:|---:|---:|']
    selected_snapshot='best' if m.get('best_metric')=='nmse' else 'last'
    for kind in m.get('groups',GROUPS):
        p=[next(r for r in rows if r['kind']==kind and r['seed']==seed and r['snapshot']==selected_snapshot
                and r['beta']==m.get('monitor_beta',.5) and r['lambda']==m.get('monitor_lambda',.1) and r['margin']=='zero')for seed in [0,1]]
        text.append(f"| {kind} | {p[0]['nmse']:.8f} | {p[1]['nmse']:.8f} | {p[0]['keep']:.4%} | {p[1]['keep']:.4%} |")
    text+=['',f'表格使用{selected_snapshot}；保留best/last及全部beta/lambda曲线。只有成本匹配与网络闭环确认后，才能判断Value增益或生成效果。',
        '初始结构及数据身份：../experiment.json；各层训练曲线与恢复点：../P3_training/；P1含128维Key参考。']
    if m.get('training_objective')=='actual_av_nmse':
        text += ['', '目标保留率约15%，仅指后续稀疏chunk；首chunk仍Dense。',
                 '主损失为真实128维注意力误差在离散硬动作分布下的期望NMSE；没有Huber主损失、软KV mask或Mask STE。',
                 'best由完整96事件验证集执行硬动作后的实际NMSE选择；延长训练也按该指标判断。',
                 '动作头输出为策略logits，不再解释成增益标量，不使用增益回归的q90余量。']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--merge',action='store_true');a=p.parse_args();root,m=config(a.directory)
    (merge if a.merge else worker)(root,m)


if __name__=='__main__':main()
