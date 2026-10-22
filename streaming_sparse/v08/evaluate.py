"""Common external replay output metrics; no teacher QKV substitution."""
import argparse
import json
from pathlib import Path
import torch
from .common import HERE,protocol,source_identity,sha,atomic_json,output,paired_seed
from .metrics import summarize,factorial
from .replay import ReplayDataset


@torch.no_grad()
def evaluate(network,ds,directory,keep=.2,mass_reference=False,mechanisms=False):
    directory=output(directory);rows=[]
    if mechanisms:(directory/'mechanisms.jsonl').write_text('')
    for index in range(len(ds)):
        sample=ds.load(index)
        result=network.replay(sample,keep=keep,checkpoint=False,collect_aux=False,recent_gradient=False,collect_mechanisms=mechanisms)
        target=sample['target'].to(result['x0'].device).float();pred=result['x0'].float()
        sse=float((pred-target).square().sum());energy=float(target.square().sum())
        diag=result['diagnostics'].cpu().tolist()
        row={k:sample[k] for k in ('sample_id','trajectory_id','prompt_id','generation_seed','history_tokens','step','timestep')}
        row.update(sse=sse,energy=energy,nmse=sse/(energy+1e-8),keep=keep,
            actual_history_keep=diag[0]/max(diag[1]*sample['history_tokens'],1),
            gate_saturated_fraction=diag[2]/max(diag[3],1),
            correction_to_base_score_l2=(diag[4]/max(diag[5],1e-30))**.5)
        if mass_reference:
            mr=network.replay(sample,keep=keep,checkpoint=False,collect_aux=False,recent_gradient=False,mass_only=True)
            row['mass_only_nmse']=float((mr['x0'].float()-target).square().sum())/(energy+1e-8)
        rows.append(row)
        if mechanisms:
            with (directory/'mechanisms.jsonl').open('a') as f:
                for record in result['mechanisms']:f.write(json.dumps(dict(sample_id=sample['sample_id'],**record))+'\n')
        if (index+1)%4==0 or index+1==len(ds):
            atomic_json(directory/'progress.json',dict(completed=index+1,total=len(ds),split=ds.split))
            print(json.dumps(dict(evaluated=index+1,total=len(ds),split=ds.split,keep=keep)),flush=True)
    summary=summarize(rows)
    atomic_json(directory/'rows.json',rows)
    atomic_json(directory/'summary.json',dict(source_sha256=source_identity(),split=ds.split,keep=keep,
        dataset_fingerprint=ds.fingerprint(),model_version=network.model_version,router_version=network.router_version,**summary))
    return summary


def load_network(path,device='cuda'):
    from .backbone import FunctionalBackbone,load_teacher
    from .router import InformationRouter
    from .lora import inject
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['source_sha256']!=source_identity():raise ValueError('Checkpoint source mismatch')
    p=protocol();wrapper=load_teacher(device);inject(wrapper.model,p['layers'],state['seed'])
    pairs=dict(wrapper.model.named_parameters())
    with torch.no_grad():
        for name,value in state['lora'].items():pairs[name].copy_(value)
    router=InformationRouter(state['information']).to(device);router.load_state_dict(state['router'])
    network=FunctionalBackbone(wrapper,router).eval()
    network.model_version=sha(path);network.router_version=network.model_version
    return network,state


def lock_registry(destination):
    # Test lock binds all three required snapshots; prevents choosing on Test.
    checkpoints={}
    for seed in range(3):
        for group in protocol()['groups']:
            directory=HERE/f'runs/seed{seed}/{group}'
            for tag in ('initial','best','last'):
                path=directory/f'{tag}.pt'
                if not path.exists():raise ValueError(f'Incomplete registry: {path}')
                state=torch.load(path,map_location='cpu',weights_only=True)
                if state['source_sha256']!=source_identity() or state['stage']!='main' or state['pilot'] or state['group']!=group or state['seed']!=seed:
                    raise ValueError('Only matching formal main checkpoints may enter the Test lock')
                if tag=='last' and state['updates']!=protocol()['main']['updates']:raise ValueError('Incomplete main training')
                checkpoints[str(path.resolve())]=sha(path)
    atomic_json(destination,dict(source_sha256=source_identity(),registry_sha256=sha(HERE/'configs/prompt_registry.json'),
        checkpoints=checkpoints,selection_split='val',test_policy='single locked evaluation, no subsequent checkpoint selection'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--keeps',type=float,nargs='+',default=[.2,.1,.3]);p.add_argument('--device',default='cuda')
    p.add_argument('--lock');p.add_argument('--make-lock');p.add_argument('--output');p.add_argument('--mass-reference',action='store_true')
    p.add_argument('--mechanisms',action='store_true')
    p.add_argument('--factorial',nargs='+',help='18 summary paths, each containing seed and group via CLI metadata')
    a=p.parse_args();paired_seed(0)
    if a.make_lock:lock_registry(a.make_lock);return
    if a.factorial:
        models={}
        for path in a.factorial:
            x=json.loads(Path(path).read_text());models[x['seed'],x['group']]=x
        atomic_json(a.output,factorial(models));return
    if not a.checkpoint or not a.output:p.error('--checkpoint and --output are required')
    ds=ReplayDataset(a.split,lock=a.lock,require_complete=True)
    if a.split=='test':
        lock=json.loads(Path(a.lock).read_text());resolved=str(Path(a.checkpoint).resolve())
        if lock['checkpoints'].get(resolved)!=sha(a.checkpoint):raise ValueError('Checkpoint not in sealed Test lock')
    from ..capture.capture_runner import sdpa_backend
    network,state=load_network(a.checkpoint,a.device)
    with sdpa_backend():
        for keep in a.keeps:
            directory=output(Path(a.output)/f'keep{int(keep*100)}')
            result=evaluate(network,ds,directory,keep,a.mass_reference,a.mechanisms)
            path=directory/'summary.json';x=json.loads(path.read_text());x.update(seed=state['seed'],group=state['group'],checkpoint_sha256=sha(a.checkpoint));atomic_json(path,x)


if __name__=='__main__':main()
