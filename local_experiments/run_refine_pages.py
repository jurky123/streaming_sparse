"""Fixed four prototypes/page at every history age; resumable controlled pilot."""
import argparse
import json
import random
import signal
import time
from pathlib import Path
import torch
from .refine_pages import make_model,prediction,loss_terms,ROOT
from .learned_pool import Bank,actions,DICTIONARY
from .learned_cache.data import records
from .learned_cache.model import predict,initial_mask,output_for_masks
from .learned_cache.run import nmse
from .learned_cache.io import atomic,checkpoint,restore,digest,disk_guard,RunLock


def identity(args):
    names=['refine_pages.py','run_refine_pages.py','page_local.py','run_page_local.py','learned_pool.py','response_hard.py','cache_grouping.py']
    files=[Path('local_experiments')/n for n in names]+list(Path('local_experiments/learned_cache').glob('*.py'))
    paths=records('train')+records('val')+records('test')
    paths=sorted(set(paths+[p.with_name(p.name[:-6]+'_history.pt') for p in paths]))
    return dict(rep=args.rep,objective=args.objective,steps=args.steps,seed=0,prototypes_per_page=4,age_schedule=False,lr=.001,
        code={str(p):digest(p) for p in files},data={str(p):digest(p) for p in paths},dictionary=digest(DICTIONARY), response_initial=digest(ROOT/'response_initial.pt'), calibration_base=digest(ROOT/'calibration_base.json') if args.rep=='calibration' else None)


@torch.no_grad()
def validation(model,bank,rep):
    scores=[]
    for p in records('val'):
        e=bank.get(p);z,_,_=prediction(model,e,rep)
        out=output_for_masks(e['z'],e['values'],initial_mask(z)[:,:,None])[:,:,0]
        scores.append(nmse(out,e['target']).mean().item())
    return sum(scores)/len(scores)


def train(args):
    root=Path(args.root);lock=RunLock(root);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.manual_seed(0);random.seed(0);torch.backends.cuda.matmul.allow_tf32=False
    model=make_model(args.rep).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    ident=identity(args);saved,name=restore(root);step=0;best=float('inf');history=[]
    if saved:
        assert saved['identity']==ident,'Resume identity mismatch'
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);step=saved['step'];best=saved['best'];history=saved['history']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'Resumed {name} step {step}',flush=True)
    bank=Bank(4);start=step;last=time.monotonic()
    def state():return dict(identity=ident,step=step,best=best,history=history,model=model.state_dict(),optimizer=opt.state_dict(),
        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    try:
        disk_guard(root.parent);atomic(root/'status.json',dict(status='running',step=step))
        if not saved:
            best=validation(model,bank,args.rep);history.append(dict(step=0,val=best));atomic(root/'best.pt',state(),True);checkpoint(root,state())
        while step<args.steps and not stop[0]:
            e=bank.get(random.choice(records('train')));opt.zero_grad(set_to_none=True)
            z,val,out=prediction(model,e,args.rep,False)
            loss=loss_terms(z,val,out,e,args.objective)
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
            if step%50==0 or step==args.steps:
                val=validation(model,bank,args.rep);history.append(dict(step=step,val=val,loss=loss.item(),gradient_norm=norm.item()))
                if val<best:best=val;atomic(root/'best.pt',state(),True)
                print(json.dumps(history[-1]),flush=True)
            planned=args.max_updates and step-start>=args.max_updates
            if step%25==0 or time.monotonic()-last>=30 or planned or stop[0] or step==args.steps:
                disk_guard(root.parent);checkpoint(root,state());last=time.monotonic();atomic(root/'status.json',dict(status='running',step=step))
            if planned:stop[0]=True
        checkpoint(root,state());atomic(root/'status.json',dict(status='completed' if step==args.steps else 'interrupted',step=step,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,disk=disk_guard(root.parent)))
    except Exception as exc:
        atomic(root/'status.json',dict(status='failed',step=step,error=repr(exc)));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--rep',choices=['random','response','calibration'],required=True)
    p.add_argument('--objective',choices=['reconstruction','split','calibration'],required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--max-updates',type=int,default=0)
    train(p.parse_args())
