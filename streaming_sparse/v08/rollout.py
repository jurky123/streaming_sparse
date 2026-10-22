"""Four original latent chunks in closed loop; no VAE or video quality claim."""
import argparse
import json
from pathlib import Path
import torch
from .common import HERE,protocol,source_identity,atomic_json,output,paired_seed,sha,nmse,save
from .backbone import FunctionalBackbone,load_teacher,native_steps
from .cache import CacheIdentity,build_history
from .replay import ReplayDataset
from .evaluate import load_network


@torch.no_grad()
def trajectory(network,embeds,pid,generation_seed,sparse,teacher_on_student=None):
    p=protocol();device=next(network.model.parameters()).device;dtype=next(network.model.parameters()).dtype
    steps,_=native_steps(network.wrapper)
    rng=torch.Generator(device=device).manual_seed(generation_seed)
    noise=torch.randn((1,12,*p['latent_shape']),generator=rng,device=device,dtype=dtype)
    # Preserve the native sampler's separate RNG calls, not one concatenated
    # random draw whose CUDA Philox offsets could differ.
    transitions=torch.stack([torch.stack([torch.randn((3,*p['latent_shape']),generator=rng,device=device,dtype=dtype)
        .unflatten(0,(1,3)) for _ in range(3)]) for _ in range(4)])
    history=noise[:,:0];chunks=[];diagnostics=[]
    for chunk in range(4):
        identity=CacheIdentity(network.model_version,network.router_version,f'{pid}.seed{generation_seed}',
                               f'rollout.chunk{chunk}',0,chunk*4680,p['context_noise'],owner='student' if sparse else 'teacher')
        cache=build_history(network,history,embeds,identity,recent_gradient=False,checkpoint=False)
        current=noise[:,chunk*3:(chunk+1)*3]
        for step,timestep in enumerate(steps):
            result=network.forward_chunk(current,embeds,timestep,cache,chunk*3,sparse=sparse,step=step,
                identity=identity,checkpoint=False,collect_aux=False)
            x0=result['x0']
            if not torch.isfinite(x0).all():raise FloatingPointError('Nonfinite closed-loop latent')
            item=dict(chunk=chunk,step=step,timestep=timestep,latent_norm=float(x0.float().norm()),
                      latent_abs_max=float(x0.float().abs().max()))
            if teacher_on_student is not None:
                sample=dict(sample_id=f'rollout.chunk{chunk}.step{step}',trajectory_id=f'{pid}.seed{generation_seed}',
                    history_tokens=chunk*4680,context_noise=p['context_noise'],step=step,timestep=timestep,
                    noisy=current,history=history,prompt_embeds=embeds)
                reference=teacher_on_student.replay(sample,sparse=False,checkpoint=False,collect_aux=False)['x0']
                item['teacher_on_student_external_state_nmse']=float(nmse(x0,reference))
            diagnostics.append(item)
            if step<3:
                current=network.wrapper.scheduler.add_noise(x0.flatten(0,1),transitions[chunk,step].flatten(0,1),
                    torch.full((3,),steps[step+1],device=device)).unflatten(0,(1,3))
        chunks.append(x0.cpu());history=torch.cat((history,x0),1)
    return chunks,diagnostics


def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--output',required=True)
    a.add_argument('--split',choices=['val','test'],default='val');a.add_argument('--lock');a.add_argument('--limit',type=int,default=4)
    a.add_argument('--device',default='cuda');a.add_argument('--teacher-on-student',action='store_true')
    args=a.parse_args();paired_seed(0)
    ds=ReplayDataset(args.split,lock=args.lock,require_complete=True)
    if args.split=='test':
        lock=json.loads(Path(args.lock).read_text())
        if lock['checkpoints'].get(str(Path(args.checkpoint).resolve()))!=sha(args.checkpoint):raise ValueError('Unsealed checkpoint')
    directory=output(args.output);student,state=load_network(args.checkpoint,args.device)
    teacher=FunctionalBackbone(load_teacher(args.device));teacher.model_version='frozen_teacher';teacher.router_version='none'
    prompts=sorted(ds.registry['splits'][args.split])[:args.limit];rows=[]
    from ..capture.capture_runner import sdpa_backend
    with sdpa_backend():
        for pid in prompts:
            for seed in protocol()['generation_seeds']:
                index=next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==seed)
                sample=ds.load(index);embeds=sample['prompt_embeds'].to(args.device,dtype=torch.bfloat16)
                tc,td=trajectory(teacher,embeds,pid,seed,False)
                sc,sd=trajectory(student,embeds,pid,seed,True,teacher if args.teacher_on_student else None)
                row=dict(prompt_id=pid,generation_seed=seed,chunks=[dict(chunk=i,latent_nmse=float(nmse(x,y)),
                    latent_frames=[i*3,(i+1)*3]) for i,(x,y) in enumerate(zip(sc,tc))],student_diagnostics=sd,teacher_diagnostics=td)
                rows.append(row);atomic_json(directory/'progress.json',dict(completed=len(rows),total=len(prompts)*2))
                save(directory/f'{pid}.seed{seed}.latents.pt',dict(student=torch.cat(sc,1),teacher=torch.cat(tc,1)))
    atomic_json(directory/'summary.json',dict(source_sha256=source_identity(),checkpoint_sha256=sha(args.checkpoint),
        rows=rows,chunk_definition='3 latent frames/chunk;4 chunks=12 latent frames; not12 original video frames',
        no_video_quality_claim=True,teacher_on_student_is_distribution_shift_diagnostic=args.teacher_on_student))


if __name__=='__main__':main()
