"""Bounded, resumable real-QKV pilot. Run with local_setup/run.sh -m ...run."""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path
import torch
from .data import Bank, records
from .model import Prototype, Corrector, predict, initial_mask, alternatives, output_for_masks
from .io import atomic, checkpoint, restore, digest, disk_guard, RunLock


def nmse(out, target):
    return (out-target).square().mean(-1)/target.square().mean(-1).clamp_min(1e-5)


def prediction(e, model, learned=True, route=True):
    h=e['hist']
    return predict(e['route' if route else 'q'],model(h['state'],h['layer'],h['heads'],learned),e['kc'],e['vc'])


def action_data(e, model, learned=True):
    z,v,_=prediction(e,model,learned)
    masks,features,gains=alternatives(z,v,initial_mask(z))
    exact=output_for_masks(e['z'],e['values'],masks)
    errors=nmse(exact,e['target'][:,:,None])
    return masks,features,gains,errors


@torch.no_grad()
def validation(bank, model, corrector, stage):
    scores=[]
    for path in records('val'):
        e=bank.get(path)
        if stage=='prototype':
            z,_,_=prediction(e,model)
            out=output_for_masks(e['z'],e['values'],initial_mask(z)[:,:,None])[:,:,0]
            score=nmse(out,e['target'])
        else:
            _,f,g,err=action_data(e,model)
            action=corrector(f,g).argmax(-1)
            score=err.gather(-1,action[:,:,None])
        scores.append(score.mean().item())
    return sum(scores)/len(scores)


def train(args):
    root=Path(args.root);lock=RunLock(root);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    model=Prototype().cuda();corrector=Corrector().cuda()
    if args.stage=='corrector':
        model.load_state_dict(torch.load(args.prototype,map_location='cuda',weights_only=False)['model'])
        model.requires_grad_(False)
    module=model if args.stage=='prototype' else corrector
    opt=torch.optim.AdamW(module.parameters(),lr=.001,weight_decay=.0001)
    paths=records('train')
    identity=dict(stage=args.stage,steps=args.steps,seed=args.seed,lr=.001,queries=64,
                  code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                  data={str(p):digest(p) for p in sorted(set(records('train')+records('val')+
                        [p.with_name(p.name[:-6]+'_history.pt') for p in records('train')+records('val')]))},
                  prototype=digest(args.prototype) if args.prototype else None)
    saved,name=restore(root);step=0;best=float('inf');history=[]
    if saved:
        if saved['identity']!=identity:raise RuntimeError('Resume identity mismatch: code, data or configuration changed')
        model.load_state_dict(saved['model']);corrector.load_state_dict(saved['corrector']);opt.load_state_dict(saved['optimizer'])
        step=saved['step'];best=saved['best'];history=saved['history']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'Resumed {name} at {step}',flush=True)
    bank=Bank(max_history=6,max_events=24);start_step=step;last=time.monotonic()
    def state():
        return dict(identity=identity,step=step,best=best,history=history,model=model.state_dict(),corrector=corrector.state_dict(),
                    optimizer=opt.state_dict(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    def status(value,**kw):
        atomic(root/'status.json',dict(status=value,step=step,pid=os.getpid(),**kw))
    try:
        disk_guard(root.parent);status('running')
        if not saved:
            best=validation(bank,model,corrector,args.stage)
            history.append(dict(step=0,val=best));atomic(root/'best.pt',state(),True);checkpoint(root,state())
        while step<args.steps and not stop[0]:
            e=bank.get(random.choice(paths));opt.zero_grad(set_to_none=True)
            if args.stage=='prototype':
                z,_,out=prediction(e,model,route=False)
                loss=nmse(out,e['target']).mean()+.01*((z-e['z']).square()*e['z'].softmax(-1)).sum(-1).mean()
            else:
                with torch.no_grad():_,f,g,errors=action_data(e,model)
                truth=errors[:,:,:1]-errors
                scores=corrector(f,g)
                loss=(scores-truth).square().mean()+.01*torch.nn.functional.cross_entropy((scores/.01).flatten(0,1),truth.argmax(-1).flatten())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
            if step%50==0 or step==args.steps:
                val=validation(bank,model,corrector,args.stage)
                history.append(dict(step=step,loss=loss.item(),val=val))
                if val<best:
                    best=val;atomic(root/'best.pt',state(),True)
                print(json.dumps(history[-1]),flush=True)
            planned=args.max_updates and step-start_step>=args.max_updates
            if step%25==0 or time.monotonic()-last>=30 or stop[0] or planned or step==args.steps:
                disk_guard(root.parent);checkpoint(root,state());last=time.monotonic();status('running')
            if planned:stop[0]=True
        checkpoint(root,state());status('completed' if step==args.steps else 'interrupted',disk=disk_guard(root.parent))
    except Exception as exc:
        status('failed',error=repr(exc));raise
    finally:lock.close()


@torch.no_grad()
def evaluate(args):
    root=Path(args.root);lock=RunLock(root)
    model=Prototype().cuda();corrector=Corrector().cuda()
    model.load_state_dict(torch.load(args.prototype,map_location='cuda',weights_only=False)['model'])
    corrector.load_state_dict(torch.load(args.corrector,map_location='cuda',weights_only=False)['corrector'])
    identity=dict(prototype=digest(args.prototype),corrector=digest(args.corrector),
                  code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text())!=identity:
        raise RuntimeError('Evaluation identity mismatch')
    atomic(root/'identity.json',identity);bank=Bank();rows=[]
    try:
        for path in records('test'):
            dest=root/(path.parent.parent.name+'_'+path.stem+'.json')
            if dest.exists():rows.extend(json.loads(dest.read_text()));continue
            disk_guard(root.parent);e=bank.get(path);batch=[]
            for learned in (False,True):
                _,f,g,err=action_data(e,model,learned)
                methods={'none':torch.zeros_like(err[:,:,0],dtype=torch.long),'analytic':g.argmax(-1),'oracle':err.argmin(-1)}
                if learned:methods['learned']=corrector(f,g).argmax(-1)
                for method,action in methods.items():
                    chosen=err.gather(-1,action[:,:,None]).squeeze(-1)
                    for hi,head in enumerate(e['hist']['heads']):
                        batch.append(dict(scene=e['case'],chunk=e['chunk'],layer=e['hist']['layer'],head=int(head),step=e['step'],
                            method=('learned' if learned else 'fixed')+'_'+method,nmse=chosen[hi].mean().item(),
                            swap=(action[hi]!=0).float().mean().item(),benefit=(chosen[hi]<err[hi,:,0]-1e-9).float().mean().item(),
                            harm=(chosen[hi]>err[hi,:,0]+1e-9).float().mean().item()))
            atomic(dest,batch);rows.extend(batch)
            print(dest.name,flush=True)
        summary={}
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['method']==method]
            summary[method]={key:sum(r[key] for r in selected)/len(selected) for key in ('nmse','swap','benefit','harm')}
        atomic(root/'summary.json',summary);atomic(root/'status.json',dict(status='completed',events=len(records('test')),disk=disk_guard(root.parent)))
        print(json.dumps(summary,indent=2),flush=True)
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['prototype','corrector','evaluate'],required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--seed',type=int,default=0);p.add_argument('--prototype');p.add_argument('--corrector');p.add_argument('--max-updates',type=int,default=0)
    args=p.parse_args();evaluate(args) if args.stage=='evaluate' else train(args)
