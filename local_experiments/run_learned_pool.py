"""Resumable learned aggregation experiment; no backbone updates."""
import argparse
import json
import random
import signal
import time
from pathlib import Path
import torch
from .learned_pool import Pool, Bank, actions, DICTIONARY
from .learned_cache.data import records
from .learned_cache.model import predict, initial_mask, output_for_masks
from .learned_cache.run import nmse
from .learned_cache.io import RunLock, atomic, checkpoint, restore, digest, disk_guard


def prediction(e,model,which='both',route=True):
    return predict(e['route' if route else 'q'],model(e['h'],which!='mean',which),e['kc'],e['vc'])


@torch.no_grad()
def validate(bank,model):
    scores=[]
    for p in records('val'):
        e=bank.get(p);z,_,_=prediction(e,model)
        out=output_for_masks(e['z'],e['values'],initial_mask(z)[:,:,None])[:,:,0]
        scores.append(nmse(out,e['target']).mean().item())
    return sum(scores)/len(scores)


def identity(args):
    files=[Path('local_experiments')/p for p in ['learned_pool.py','run_learned_pool.py','response_hard.py','cache_grouping.py']]
    files+=list(Path('local_experiments/learned_cache').glob('*.py'))
    data=records('train')+records('val')+records('test')
    data=sorted(set(data+[p.with_name(p.name[:-6]+'_history.pt') for p in data]))
    return dict(prototypes=args.prototypes,steps=args.steps,seed=0,lr=.001,
        sources={str(p):digest(p) for p in files},data={str(p):digest(p) for p in data},dictionary=digest(DICTIONARY))


def train(args):
    root=Path(args.root);lock=RunLock(root);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.manual_seed(0);random.seed(0);torch.backends.cuda.matmul.allow_tf32=False
    model=Pool().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    ident=identity(args);saved,name=restore(root);step=0;best=float('inf');history=[]
    if saved:
        assert saved['identity']==ident,'Resume code/data/config mismatch'
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);step=saved['step'];best=saved['best'];history=saved['history']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'Resumed {name} step {step}',flush=True)
    bank=Bank(args.prototypes);start=step;last=time.monotonic()
    def state():
        return dict(identity=ident,step=step,best=best,history=history,model=model.state_dict(),optimizer=opt.state_dict(),
                    python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    try:
        disk_guard(root.parent);atomic(root/'status.json',dict(status='running',step=step))
        if not saved:
            best=validate(bank,model);history.append(dict(step=0,val=best));atomic(root/'best.pt',state(),True);checkpoint(root,state())
        while step<args.steps and not stop[0]:
            e=bank.get(random.choice(records('train')));opt.zero_grad(set_to_none=True)
            z,val,out=prediction(e,model,route=False)
            mass=e['z'].softmax(-1)
            full=nmse(out,e['target']).mean()
            part=((z-e['z']).square()*mass).sum(-1).mean()
            page=((val-e['values']).square().mean(-1)*mass).sum(-1)/e['target'].square().mean(-1).clamp_min(1e-5)
            loss=full+.01*part+.1*page.mean()
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
            if step%50==0 or step==args.steps:
                score=validate(bank,model);history.append(dict(step=step,val=score,loss=loss.item(),full=full.item(),page=page.mean().item(),partition=part.item()))
                if score<best:best=score;atomic(root/'best.pt',state(),True)
                print(json.dumps(history[-1]),flush=True)
            planned=args.max_updates and step-start>=args.max_updates
            if step%25==0 or time.monotonic()-last>=30 or stop[0] or planned or step==args.steps:
                disk_guard(root.parent);checkpoint(root,state());last=time.monotonic()
                atomic(root/'status.json',dict(status='running',step=step))
            if planned:stop[0]=True
        checkpoint(root,state());atomic(root/'status.json',dict(status='completed' if step==args.steps else 'interrupted',step=step,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,disk=disk_guard(root.parent)))
    except Exception as exc:
        atomic(root/'status.json',dict(status='failed',step=step,error=repr(exc)));raise
    finally:lock.close()


@torch.no_grad()
def evaluate(args):
    root=Path(args.root);dest=root/'evaluation';lock=RunLock(dest);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    saved=torch.load(root/'best.pt',weights_only=False,map_location='cpu');assert saved['identity']==identity(args)
    ident=dict(checkpoint=digest(root/'best.pt'),identity=saved['identity'])
    if (dest/'identity.json').exists():assert json.loads((dest/'identity.json').read_text())==ident
    else:atomic(dest/'identity.json',ident)
    model=Pool().cuda();model.load_state_dict(saved['model']);bank=Bank(args.prototypes)
    try:
        for path in records('test'):
            shard=dest/(path.parent.parent.name+'_'+path.stem+'.json')
            if shard.exists():continue
            if stop[0]:break
            disk_guard(root.parent);e=bank.get(path);rows=[]
            for which in ['mean','k','v','both']:
                z,val,_=prediction(e,model,which)
                for kind in ['boundary','residual']:
                    masks,action,proxy=actions(z,val,kind)
                    outputs=output_for_masks(e['z'],e['values'],masks)
                    errors=nmse(outputs,e['target'][:,:,None])
                    modes={kind:action,kind+'_oracle':errors.argmin(-1)}
                    if kind=='boundary':modes['none']=torch.zeros_like(action)
                    for mode,index in modes.items():
                        chosen=errors.gather(-1,index[:,:,None]).squeeze(-1)
                        for hi,head in enumerate(e['h']['heads']):
                            rows.append(dict(scene=e['scene'],layer=e['h']['layer'],head=int(head),chunk=e['chunk'],step=e['step'],
                                pool=which,correction=mode,nmse=chosen[hi].mean().item(),swap=(index[hi]!=0).float().mean().item(),
                                benefit=(chosen[hi]<errors[hi,:,0]-1e-9).float().mean().item(),harm=(chosen[hi]>errors[hi,:,0]+1e-9).float().mean().item()))
            atomic(shard,rows);print(shard.name,flush=True)
        rows=[]
        for shard in sorted(dest.glob('*_s?.json')):rows.extend(json.loads(shard.read_text()))
        summary={}
        for which in ['mean','k','v','both']:
            for kind in ['none','boundary','residual','boundary_oracle','residual_oracle']:
                rr=[r for r in rows if r['pool']==which and r['correction']==kind]
                if rr:summary[which+'_'+kind]={key:sum(r[key] for r in rr)/len(rr) for key in ['nmse','swap','benefit','harm']}
        atomic(dest/'summary.json',summary);atomic(dest/'status.json',dict(status='interrupted' if stop[0] else 'completed',events=len(rows)//60,disk=disk_guard(root.parent)))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        atomic(dest/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--prototypes',type=int,choices=[1,4],required=True)
    parser.add_argument('--steps',type=int,default=600);parser.add_argument('--max-updates',type=int,default=0);parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();evaluate(args) if args.evaluate else train(args)
