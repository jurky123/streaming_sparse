"""Train-only calibration and a separate true-AV 15% action-head experiment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import torch

from .common import ROOT, config, atomic_json, load_raw, update, sha
from .core import build, scores, initial_mask


def prepare(parent, root):
    parent, source = config(parent)
    assert json.loads((parent/'status.json').read_text())['status']=='phase_complete'
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    m=dict(source)
    m.update(directory=str(root),created_at=datetime.now(timezone.utc).isoformat(),
        version='SlimAV-v0.10-keep15-actualAV-loss',parent_experiment=str(parent),
        feature_namespace=root.name,training_objective='actual_av_nmse',
        objective_formula='mean_event sum_head_query_action softmax(logits/T) * exact_hard_action_SSE / dense_energy_event',
        action_head_output='policy logits, not predicted gains',target_keep=.15,
        target_keep_scope='Macro mean over nonfirst sparse events; first chunk stays Dense and is excluded.',
        budget_calibration='48 balanced Train events per layer, 32 fixed Queries; global beta chosen before Val evaluation.',
        beta_grid=[1.,1.1,1.2],train_betas=[1.,1.1,1.2],lambda_grid=[0.],
        monitor_beta=1.1,monitor_lambda=0.,monitor_events='all96',report_margins=['zero'],
        acceptance_margins='Zero only; policy logits are not calibrated gain predictions.',
        temperature_schedule='1.0 to0.25 linearly in first400 updates; then0.25',
        best_metric='nmse',best_checkpoint_criterion='Minimum actual hard-action attention NMSE on all96 nonfirst Val events; same fixed beta and lambda0.',
        P3_extension_rule='At400 updates, extend all8 runs for a layer to800 if any improves actual hard-action Val NMSE >=0.5% in its last3 validations over prior best.',
        P3_monitor='All96 nonfirst Val events at the Train-calibrated target beta and lambda0, every50 updates.',
        baseline_modes=['Dense','U-Sol'],
        foundation_reused_from=str(parent),
        action_basis_initialization='Fresh low-budget Train action-error covariance; no inherited action head.',
        min_updates=400,max_updates=800)
    m['user_amendments']=[*m['user_amendments'],'Target approximately15% KV in sparse chunks',
                           'Optimize actual attention error; select best by actual hard-action attention error']
    atomic_json(root/'experiment.json',m)
    # Raw data and K-only parameters are unchanged; Value basis and heads are new.
    (root/'parameters').symlink_to(parent/'parameters',target_is_directory=True)
    (root/'capture_workers').symlink_to(parent/'capture_workers',target_is_directory=True)
    atomic_json(root/'foundation_inheritance.json',dict(parent=str(parent),
        parameters={p.name:sha(p)for p in (parent/'parameters').glob('layer*.pt')},
        raw_cache=m['cache'],action_heads_inherited=False,value_basis_inherited=False))
    atomic_json(root/'status.json',dict(status='prepared',stage='B15_calibration'))
    (root/'README.txt').write_text(
        '目标：后续稀疏chunk约15% KV；首chunk Dense。历史原型P8/P4/P2、当前P16。\n'
        'K-only基础参数与原始采集复用；低预算Value投影、动作特征与动作头重新生成和训练。\n'
        '主损失：原始128维KV执行各硬动作产生的真实注意力NMSE，在动作概率下的期望。\n'
        'best：完整96事件验证集执行确定硬动作后的实际注意力NMSE最低；不按Huber选择。\n'
        '先用Train校准全局beta；保留率统计排除首chunk，并分别报告历史与当前保留率。\n'
        '四组×两个种子×六层，400步起；按实际NMSE配对决定是否延长到800步。\n'
        '查看：./v10_progress.sh；完整快照：./v10_progress.sh --once --details\n')
    atomic_json(ROOT/'migration/unified_router_active.json',dict(directory=str(root)))
    print(json.dumps(dict(directory=str(root),target_keep=.15,loss='actual_av_nmse')),flush=True)


@torch.no_grad()
def calibrate(root,m):
    torch.cuda.set_device(0)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    betas=[round(.7+i*.01,2)for i in range(81)]
    train=[r for r in m['rows']if r['split']=='train'and r['history_tokens']]
    chosen=[next(r for r in train if r['prompt_id']==pid and r['history_tokens']==history
                and r['generation_seed']==(i+j)%2 and r['step']==(i+j)%4)
            for i,pid in enumerate(m['splits']['train'])for j,history in enumerate([4680,9360,14040])]
    for layer in m['layers'][rank::world]:
        path=root/f'B15_calibration/layer{layer:02d}.json'
        if path.is_file():continue
        parameters=torch.load(root/f'parameters/layer{layer:02d}.pt',map_location='cpu',weights_only=True)['parameters']
        parameters={k:v.cuda()for k,v in parameters.items()}
        totals=[0.]*len(betas)
        for i,row in enumerate(chosen,1):
            d=load_raw(root,row,layer);d['q']=d['q'][:,:32]
            sketch=build(d['kh'],d['vh'],d['kc'],d['vc'],parameters,need_value=False)
            score,_=scores(d['q'],sketch,content=None)
            for j,beta in enumerate(betas):
                mask=initial_mask(score,sketch,beta)
                totals[j]+=float((mask*sketch.costs[None,None]).sum(-1).mean()/(row['history_tokens']+4680))
            update(root/f'B15_calibration/rank{rank}.json',status='evaluating',layer=layer,completed=i,total=len(chosen))
        atomic_json(path,dict(layer=layer,betas=betas,keep=[x/len(chosen)for x in totals],
            split='train',sample_ids=[r['sample_id']for r in chosen],queries_per_event=32))
    update(root/f'B15_calibration/rank{rank}.json',status='completed',layers=m['layers'][rank::world])


def lock(root,m):
    rows=[json.loads((root/f'B15_calibration/layer{layer:02d}.json').read_text())for layer in m['layers']]
    betas=rows[0]['betas']
    mean=[sum(r['keep'][i]for r in rows)/len(rows)for i in range(len(betas))]
    index=min(range(len(betas)),key=lambda i:abs(mean[i]-m['target_keep']))
    beta=betas[index]
    assert abs(mean[index]-m['target_keep'])<.005,(beta,mean[index])
    m.update(monitor_beta=beta,beta_grid=[round(beta-.08,4),beta,round(beta+.08,4)],
             train_betas=[round(beta-.08,4),beta,round(beta+.08,4)],
             validation_betas=[round(beta-.08,4),beta,round(beta+.08,4)],budget_locked=True)
    atomic_json(root/'experiment.json',m)
    atomic_json(root/'budget_calibration.json',dict(target_keep=m['target_keep'],beta=beta,
        observed_train_keep=mean[index],split='train',val_used=False,
        per_layer={str(r['layer']):r['keep'][index]for r in rows},
        curve=[dict(beta=b,keep=k)for b,k in zip(betas,mean)]))
    print(json.dumps(dict(beta=beta,train_keep=mean[index],target=.15)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True)
    p.add_argument('--prepare-from');p.add_argument('--lock',action='store_true');a=p.parse_args()
    if a.prepare_from:
        prepare(a.prepare_from,a.directory)
    else:
        root,m=config(a.directory)
        (lock if a.lock else calibrate)(root,m)


if __name__=='__main__':main()
