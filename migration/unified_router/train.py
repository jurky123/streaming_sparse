"""Four paired action heads x two seeds; protocol controls loss and selection."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import torch

from .common import config, save, atomic_json, update, paired_seed, sha
from .labels import GROUPS, train_betas
from .learning import ActionHead,batch,objective,evaluate


def paired_cap(root,layer,kind,seed,curve,metric='huber',groups=None,seeds=(0,1)):
    directory=root/f'P3_training/layer{layer:02d}/extension'
    old=min(r[metric]for r in curve[:-3])
    recent=min(r[metric]for r in curve[-3:])
    improve=(old-recent)/max(old,1e-8)>=.005
    atomic_json(directory/f'{kind}.seed{seed}.json',dict(improving=improve,old=old,recent=recent,metric=metric))
    paths=[directory/f'{g}.seed{s}.json'for g in (groups or GROUPS) for s in seeds]
    started=time.monotonic()
    while not all(p.is_file()for p in paths):
        if(root/'STOP').exists():raise RuntimeError('Stop requested')
        if time.monotonic()-started>6*3600:raise TimeoutError('Paired400-update decision did not arrive')
        time.sleep(3)
    return 800 if any(json.loads(p.read_text())['improving'] for p in paths) else 400


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args()
    root,m=config(a.directory);torch.cuda.set_device(0)
    assert json.loads((root/'P3_preflight.json').read_text())['passed']
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    groups=m.get('groups',list(GROUPS));seeds=m.get('training_seeds',[0,1])
    teams=m.get('train_layer_teams',1);team_size=len(groups)*len(seeds)
    assert world==team_size*teams,'Worker count must match the predeclared paired teams'
    team=rank//team_size;local=rank%team_size
    seed=seeds[local//len(groups)];kind=groups[local%len(groups)]
    train=[r for r in m['rows']if r['split']=='train'and r['history_tokens']]
    val=[r for r in m['rows']if r['split']=='val'and r['history_tokens']]
    pids=m['splits']['val']
    monitor=[r for r in val if r['step']==(pids.index(r['prompt_id'])+r['generation_seed']+r['history_tokens']//4680-1)%4]
    assert len(monitor)==24
    if m.get('monitor_events') == 'all96':
        monitor=val
    metric=m.get('best_metric','huber');monitor_beta=m.get('monitor_beta',.5)
    monitor_lambda=m.get('monitor_lambda',.1)
    diagnostic='expected_nmse' if m.get('training_objective')=='actual_av_nmse' else 'huber'
    my_layers=m['layers'][team::teams]
    for layer in my_layers:
        directory=root/f'P3_training/layer{layer:02d}/{kind}/seed{seed}'
        directory.mkdir(parents=True,exist_ok=True)
        if(directory/'completed.json').exists():continue
        paired_seed(seed*1000+layer)
        normal=torch.load(root/f'normalization/layer{layer:02d}.pt',map_location='cpu',weights_only=True)[kind]
        basis=torch.load(root/f'action_basis/layer{layer:02d}.pt',map_location='cpu',weights_only=True)
        model=ActionHead(normal).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=m['learning_rate'],weight_decay=.01)
        rng=random.Random(10000+seed);pool=[];curve=[];begin=0;cap=400;best=math.inf
        recovery=directory/'recovery.pt'
        if recovery.exists():
            d=torch.load(recovery,map_location='cpu',weights_only=True)
            model.load_state_dict(d['head']);optimizer.load_state_dict(d['optimizer'])
            rng.setstate(d['rng']);pool=d['pool'];curve=d['curve'];begin=d['updates'];cap=d['cap'];best=d['best']
        else:
            initial=evaluate(model,root,m,monitor,layer,monitor_beta,kind,basis,monitor_lambda)
            curve.append(dict(updates=0,**{k:v for k,v in initial.items()if k!='rows'}));best=initial[metric]
            save(directory/'initial.pt',dict(head=model.state_dict(),updates=0,kind=kind,seed=seed,layer=layer))
            save(directory/'best.pt',dict(head=model.state_dict(),updates=0,kind=kind,seed=seed,layer=layer,best_metric=metric,best_value=best))
            atomic_json(directory/'best_validation.json',dict(updates=0,beta=monitor_beta,**initial))
        def checkpoint(step):
            save(recovery,dict(head=model.state_dict(),optimizer=optimizer.state_dict(),updates=step,
                rng=rng.getstate(),pool=pool,curve=curve,best=best,cap=cap,kind=kind,seed=seed,layer=layer))
        checkpoint(begin)
        step=begin
        while step<cap:
            if(root/'STOP').exists():raise RuntimeError('Stop requested')
            step+=1;chosen=[]
            for _ in range(8):
                if not pool:pool=list(range(len(train)));rng.shuffle(pool)
                chosen.append(train[pool.pop()])
            betas=train_betas(m)
            beta=betas[(step-1)%len(betas)];lam=m['lambda_grid'][(step-1)%len(m['lambda_grid'])]
            b=batch(root,m,chosen,layer,beta,kind,basis)
            optimizer.zero_grad(set_to_none=True)
            pred=model(b['x']);loss,huber=objective(pred,b,lam,1.-.75*min(step/400,1.))
            assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if step==begin+1:assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
            lr=m['learning_rate']*min(1.,step/20)*(.05+.95*.5*(1+math.cos(math.pi*step/800)))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step()
            if step%10==0 or step==begin+1:
                update(root/f'P3_training/rank{rank}.json',status='training',kind=kind,seed=seed,layer=layer,
                    updates=step,cap=cap,loss=float(loss),**{diagnostic:float(huber)},gradient_norm=float(grad),beta=beta,learning_rate=lr,best_metric=metric)
            if step%50==0:
                result=evaluate(model,root,m,monitor,layer,monitor_beta,kind,basis,monitor_lambda)
                curve.append(dict(updates=step,**{k:v for k,v in result.items()if k!='rows'}))
                atomic_json(directory/'validation.json',dict(curve=curve,kind=kind,seed=seed,layer=layer))
                if result[metric]<best:
                    best=result[metric];save(directory/'best.pt',dict(head=model.state_dict(),updates=step,kind=kind,seed=seed,layer=layer,best_metric=metric,best_value=best))
                    atomic_json(directory/'best_validation.json',dict(updates=step,beta=monitor_beta,**result))
                if step==400:
                    update(root/f'P3_training/rank{rank}.json',status='paired_extension_wait',kind=kind,seed=seed,layer=layer,updates=step)
                    cap=paired_cap(root,layer,kind,seed,curve,metric,groups,seeds)
            if step%25==0:checkpoint(step)
        save(directory/'last.pt',dict(head=model.state_dict(),updates=step,kind=kind,seed=seed,layer=layer))
        checkpoint(step)
        result=evaluate(model,root,m,val,layer,monitor_beta,kind,basis,monitor_lambda)
        atomic_json(directory/'final_validation.json',result)
        atomic_json(directory/'completed.json',dict(updates=step,kind=kind,seed=seed,layer=layer,
            head_sha256=sha(directory/'last.pt'),backbone_loaded=False,backbone_and_lora_frozen=True))
    update(root/f'P3_training/rank{rank}.json',status='completed',kind=kind,seed=seed,layers=my_layers)


if __name__=='__main__':main()
