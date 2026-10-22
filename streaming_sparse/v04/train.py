"""Six paired runs: two fixed train budgets x three seeds; final update6000 is primary."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import sys
import torch
from .data import SnapshotData,BalancedSchedule
from .identity import source_identity
from .model import PairedRouter,RouterConfig
from ..v03.evaluate import statistics
from ..v03.metrics import error_parts
from .paths import protect_output
from ..capture.dataset_io import query_slices,atomic_save
from ..v02.evaluation_journal import atomic_json


def train_update(model,batch,keep,optimizer):
    """batch contains (four aligned events, shared tile index) for each snapshot."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals=dict(A=0.,B=0.)
    samples=[]
    for events,tile in batch:
        e0=events[0]
        state=model.precompute(e0['k_hist'],e0['v_hist'],head_id=e0['head_id'],layer_id=e0['layer_id'])
        losses={name:[] for name in ('A','B')}
        reference=None
        for event in events:
            selection=query_slices(event,16)[tile]
            ids=event['query_indices'][selection]
            if reference is None:reference=ids
            elif not torch.equal(ids,reference):raise ValueError('Misaligned diffusion-step tiles')
            q=event['q'][selection]
            stat=statistics(q,event,state['assignment'])
            routes=model.route(q,state,keep)
            for name,route in routes.items():
                se,en=error_parts(stat.output(route['mask']),event['o_dense'][selection])
                losses[name].append(se/(en+1e-8))
        arm_losses={name:torch.stack(values).mean() for name,values in losses.items()}
        joint=.5*(arm_losses['A']+arm_losses['B'])/len(batch)
        if not torch.isfinite(joint):raise FloatingPointError('Nonfinite paired attention loss')
        joint.backward()
        for name,value in arm_losses.items():totals[name]+=float(value.detach())/len(batch)
        samples.append(dict(cache_version=e0.get('cache_version','synthetic'),tile_id=tile,
                            partition_hash_A=state['partition_hash'],partition_hash_B=state['partition_hash']))
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    optimizer.step()
    return dict(attention_loss=totals,joint_loss=.5*(totals['A']+totals['B']),gradient_norm=norm,
                samples=samples,keep=keep)


def configuration(keep,seed):
    if keep not in (.3,.2) or seed not in (0,1,2):
        raise ValueError('Separate fixed budgets .3/.2 and router seeds0/1/2')
    return dict(steps=6000,snapshots_per_update=4,tiles_per_event=1,diffusion_steps_per_snapshot=4,
        lr=1e-4,weight_decay=.01,grad_clip=1.,checkpoint_every=1000,precision='bf16 fusion/interaction; fp32 frontend,score,stats,gate,loss',
        train_keep=keep,router_seed=seed,frontend_training='shared trainable, equal mean of A/B attention losses',
        checkpoint_selection='fixed paired final update6000',loss='normalized attention SSE only',
        sampler='without-replacement snapshots; rotated eight tiles; shared A/B samples')


def run(data_root,output,keep,seed,sanity_gate,resume=None):
    config=configuration(keep,seed)
    current_source=source_identity()
    gate=json.loads(Path(sanity_gate).read_text())
    if (not gate.get('passed') or not gate.get('real_qkv') or not gate.get('full_head_coverage') or
        not gate.get('bf16_checked') or gate.get('source_sha256')!=current_source):
        raise ValueError('Current-source real BF16/full12-head correctness gate required')
    data=SnapshotData(data_root,'train')
    for split in ('val','test'):
        SnapshotData(data_root,split)  # Metadata/reference check only; no Test tensor reads.
    if gate.get('data_fingerprint')!=data.fingerprint():
        raise ValueError('Sanity gate belongs to different QKV data')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Expose one BF16-capable GPU')
    if gate.get('gpu_name')!=torch.cuda.get_device_name(0) or gate.get('torch')!=str(torch.__version__):
        raise ValueError('Sanity gate belongs to different GPU hardware or Torch version')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    random.seed(seed);torch.manual_seed(seed)
    identity=dict(version='0.4',source_sha256=current_source,config=config,data_fingerprint=data.fingerprint(),
        torch=str(torch.__version__),cuda_runtime=torch.version.cuda,python=sys.version,
        gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        precision=config['precision'])
    output=protect_output(output)
    if resume and Path(resume).resolve().parent!=output:
        raise ValueError('Resume must use its original run directory')
    if output.exists() and any(output.iterdir()) and not resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    model=PairedRouter().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    sampler=BalancedSchedule(data.groups,seed)
    begin=0
    if resume:
        value=torch.load(resume,map_location='cuda',weights_only=True)
        if value['identity']!=identity:raise ValueError('Resume config/source/data/environment mismatch')
        model.load_state_dict(value['router']);optimizer.load_state_dict(value['optimizer'])
        sampler.load_state_dict(value['sampler'])
        begin=value['updates']
        torch.set_rng_state(value['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda_rng']])
        random.setstate(value['python_rng'])
        path=output/'metrics.jsonl'
        if path.exists():
            rows=[json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['updates']<=begin))
    atomic_json(output/'provenance.json',identity)

    def save(path,updates):
        atomic_save(dict(**model.checkpoint(),identity=identity,optimizer=optimizer.state_dict(),
            sampler=sampler.state_dict(),updates=updates,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),path)

    if not resume:save(output/'initial.pt',0)
    for step in range(begin,6000):
        chosen=sampler.sample(4)
        batch=[(data.load(v,'cuda'),tile) for v,tile in chosen]
        row=dict(updates=step+1,**train_update(model,batch,keep,optimizer))
        with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if step==0 or (step+1)%50==0:
            print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
        if (step+1)%1000==0:
            save(output/f'step{step+1:06d}.pt',step+1)
            save(output/'latest.pt',step+1)
            atomic_json(output/'status.json',dict(status='training' if step+1<6000 else 'completed',
                updates=step+1,train_keep=keep,router_seed=seed,updated_utc=datetime.now(timezone.utc).isoformat()))
    save(output/'final.pt',6000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--keep',type=float,choices=[.3,.2],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--sanity-gate',required=True);p.add_argument('--resume')
    args=p.parse_args()
    run(args.data,args.output,args.keep,args.seed,args.sanity_gate,args.resume)


if __name__=='__main__':main()
