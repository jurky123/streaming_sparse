"""Fixed four prototypes/page at every history age; resumable controlled pilot."""
import argparse
import json
import random
import signal
import time
from pathlib import Path
import torch
from .page_local import make_model,prototypes,hard_output
from .learned_pool import Bank,actions,DICTIONARY
from .learned_cache.data import records
from .learned_cache.model import predict,initial_mask,output_for_masks
from .learned_cache.run import nmse
from .learned_cache.io import atomic,checkpoint,restore,digest,disk_guard,RunLock

ROOT=Path('artifacts/page_local_20261030')


def prediction(model,e,rep,route=True):
    return predict(e['route' if route else 'q'],prototypes(model,e['h'],rep),e['kc'],e['vc'])


def identity(args):
    names=['page_local.py','run_page_local.py','learned_pool.py','response_hard.py','cache_grouping.py']
    files=[Path('local_experiments')/n for n in names]+list(Path('local_experiments/learned_cache').glob('*.py'))
    paths=records('train')+records('val')+records('test')
    paths=sorted(set(paths+[p.with_name(p.name[:-6]+'_history.pt') for p in paths]))
    return dict(rep=args.rep,objective=args.objective,steps=args.steps,seed=0,prototypes_per_page=4,age_schedule=False,lr=.001,
        code={str(p):digest(p) for p in files},data={str(p):digest(p) for p in paths},dictionary=digest(DICTIONARY))


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
            if args.objective=='hard':
                z,_,_=prediction(model,e,args.rep);out=hard_output(z,e)
                loss=nmse(out,e['target']).mean()
            else:
                z,val,out=prediction(model,e,args.rep,False);mass=e['z'].softmax(-1)
                page=((val-e['values']).square().mean(-1)*mass).sum(-1)/e['target'].square().mean(-1).clamp_min(1e-5)
                loss=nmse(out,e['target']).mean()+.01*((z-e['z']).square()*mass).sum(-1).mean()+.1*page.mean()
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


@torch.no_grad()
def evaluate(args):
    root=Path(args.root);dest=root/'evaluation';lock=RunLock(dest);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0)
    models={'mixture_initial':(make_model('mixture').cuda(),'mixture')};hashes={}
    for name,rep in [('mixture_reconstruction','mixture'),('mixture_hard','mixture'),('pool_hard','pool')]:
        path=root/name/'best.pt';state=torch.load(path,weights_only=False,map_location='cpu')
        assert json.loads((path.parent/'status.json').read_text())['status']=='completed'
        for source,sha in state['identity']['code'].items():assert digest(source)==sha
        model=make_model(rep).cuda();model.load_state_dict(state['model']);models[name]=(model,rep);hashes[name]=digest(path)
    prior=Path('artifacts/learned_pool_20261030/p4/best.pt');model=make_model('pool').cuda()
    model.load_state_dict(torch.load(prior,weights_only=False,map_location='cpu')['model']);models['pool_reconstruction']=(model,'pool');hashes['pool_reconstruction']=digest(prior)
    models['hard_mean']=(model,'mean')
    current_identity=dict(checkpoints=hashes,code={str(Path(__file__)):digest(__file__)},dictionary=digest(DICTIONARY))
    if (dest/'identity.json').exists():assert json.loads((dest/'identity.json').read_text())==current_identity
    else:atomic(dest/'identity.json',current_identity)
    bank=Bank(4)
    try:
        for path in records('test'):
            shard=dest/(path.parent.parent.name+'_'+path.stem+'.json')
            if shard.exists():continue
            if stop[0]:break
            disk_guard(root);e=bank.get(path);rows=[]
            for name,(model,rep) in models.items():
                proto=model(e['h'],False) if rep=='mean' else prototypes(model,e['h'],rep)
                z,val,_=predict(e['route'],proto,e['kc'],e['vc'])
                for kind in ['boundary','residual']:
                    masks,action,_=actions(z,val,kind);out=output_for_masks(e['z'],e['values'],masks)
                    errors=nmse(out,e['target'][:,:,None]);modes={kind:action,kind+'_oracle':errors.argmin(-1)}
                    if kind=='boundary':modes['none']=torch.zeros_like(action)
                    for mode,index in modes.items():
                        chosen=errors.gather(-1,index[:,:,None]).squeeze(-1)
                        for hi,head in enumerate(e['h']['heads']):
                            rows.append(dict(model=name,correction=mode,scene=e['scene'],layer=e['h']['layer'],head=int(head),chunk=e['chunk'],step=e['step'],
                                nmse=chosen[hi].mean().item(),swap=(index[hi]!=0).float().mean().item(),
                                harm=(chosen[hi]>errors[hi,:,0]+1e-9).float().mean().item(),benefit=(chosen[hi]<errors[hi,:,0]-1e-9).float().mean().item()))
            atomic(shard,rows);print(shard.name,flush=True)
        rows=[]
        for shard in dest.glob('*_s?.json'):rows.extend(json.loads(shard.read_text()))
        summary={}
        for name,mode in sorted({(r['model'],r['correction']) for r in rows}):
            rr=[r for r in rows if r['model']==name and r['correction']==mode]
            summary[name+'_'+mode]={k:sum(r[k] for r in rr)/len(rr) for k in ['nmse','swap','harm','benefit']}
        atomic(dest/'summary.json',summary);atomic(dest/'status.json',dict(status='interrupted' if stop[0] else 'completed',events=len(rows)//90,disk=disk_guard(root)))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        atomic(dest/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--rep',choices=['mixture','pool'],default='mixture')
    p.add_argument('--objective',choices=['hard','reconstruction'],default='hard');p.add_argument('--steps',type=int,default=600)
    p.add_argument('--max-updates',type=int,default=0);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    evaluate(a) if a.evaluate else train(a)
