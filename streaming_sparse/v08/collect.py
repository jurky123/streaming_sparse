"""Record native dense trajectories without VAE, not guessed noisy latents."""
import argparse
import hashlib
import json
import fcntl
from pathlib import Path
import torch
from .common import HERE,protocol,paired_seed,save,sha,atomic_json,source_identity,output
from .replay import prepare_registry,inventory
from .backbone import load_teacher,native_steps,native_cache


def collect(split,limit=None,seeds=(0,1),device='cuda',offset=0):
    from utils.wan_wrapper import WanTextEncoder
    from ..capture.capture_runner import sdpa_backend
    p=protocol();registry=prepare_registry();root=output(HERE/'data/replay')
    # Capture is allowed for sealed Test; ReplayDataset/evaluation remain locked.
    prompts=registry['splits'][split][offset:None if limit is None else offset+limit]
    embedding_dir=output(HERE/'data/embeddings')
    missing=[i for i in prompts if not (embedding_dir/f'{i}.pt').exists()]
    if missing:
        encoder=WanTextEncoder().to(device=device,dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for i in missing:
                value=encoder([registry['texts'][i]])['prompt_embeds'].cpu()
                save(embedding_dir/f'{i}.pt',dict(prompt_id=i,prompt=registry['texts'][i],prompt_embeds=value))
        del encoder;torch.cuda.empty_cache()
    wrapper=load_teacher(device);steps,config=native_steps(wrapper)
    scheduler_id=hashlib.sha256(json.dumps(dict(config=config,steps=steps),sort_keys=True).encode()).hexdigest()
    manifest=root/'events.jsonl'
    existing=[json.loads(s) for s in manifest.read_text().splitlines()] if manifest.exists() else []
    for pid in prompts:
        embed=torch.load(embedding_dir/f'{pid}.pt',weights_only=True,map_location='cpu')['prompt_embeds']
        for generation_seed in seeds:
            trajectory=f'{pid}.seed{generation_seed}'
            target=root/f'{trajectory}.pt'
            matching=[r for r in existing if r['trajectory_id']==trajectory]
            if target.exists() and len(matching)==12 and all(r['payload_sha256']==sha(target) for r in matching):
                continue
            if target.exists() or matching:
                raise ValueError('Partial/conflicting trajectory; preserve and resolve before recapture')
            paired_seed(generation_seed)
            noise=torch.randn((1,12,*p['latent_shape']),device=device,dtype=torch.bfloat16)
            cache=native_cache(wrapper,device,noise.dtype)
            cross=[None]*len(cache);latents=[];noisy=[];teacher=[]
            with torch.no_grad(),sdpa_backend():
                for chunk in range(4):
                    current=noise[:,chunk*3:(chunk+1)*3];states=[];outputs=[]
                    for step,t in enumerate(steps):
                        states.append(current.cpu().clone())
                        timestep=torch.full((1,3),t,device=device,dtype=torch.float32)
                        _,x0=wrapper(current,{'prompt_embeds':embed.to(device)},timestep,
                            kv_cache=cache,crossattn_cache=cross,current_start=chunk*3*p['frame_tokens'])
                        outputs.append(x0.cpu().clone())
                        if step<3:
                            current=wrapper.scheduler.add_noise(x0.flatten(0,1),torch.randn_like(x0.flatten(0,1)),
                                torch.full((3,),steps[step+1],device=device)).unflatten(0,(1,3))
                    latents.append(x0.cpu().clone());noisy.append(torch.stack(states));teacher.append(torch.stack(outputs))
                    wrapper(x0,{'prompt_embeds':embed.to(device)},torch.full((1,3),p['context_noise'],device=device),
                            kv_cache=cache,crossattn_cache=cross,current_start=chunk*3*p['frame_tokens'])
            payload=dict(trajectory_id=trajectory,prompt_id=pid,prompt=registry['texts'][pid],generation_seed=generation_seed,
                prompt_embeds=embed,latents=torch.cat(latents,1),noisy=torch.stack(noisy),teacher_x0=torch.stack(teacher),
                teacher_sha256=p['teacher_sha256'],scheduler_identity=scheduler_id,native_timesteps=steps,
                scheduler_config=config,context_noise=p['context_noise'],cfg=p['cfg'],parameterization='x0',
                latent_frame_definition='original3 latent frames per chunk, 1560 tokens/frame',source_sha256=source_identity())
            payload['embedding_sha256']=sha(embedding_dir/f'{pid}.pt')
            save(target,payload);digest=sha(target)
            rows=[dict(prompt_id=pid,split=split,generation_seed=generation_seed,trajectory_id=trajectory,
                sample_id=f'{trajectory}.chunk{chunk}.step{step}',chunk=chunk,step=step,timestep=steps[step],
                history_tokens=chunk*3*p['frame_tokens'],context_noise=p['context_noise'],
                history_positions=[0,chunk*3*p['frame_tokens']],teacher_sha256=p['teacher_sha256'],
                parameterization='x0',scheduler_identity=scheduler_id,payload=target.name,payload_sha256=digest)
                for chunk in (1,2,3) for step in range(4)]
            with (root/'manifest.lock').open('a') as guard:
                fcntl.flock(guard,fcntl.LOCK_EX)
                existing=[json.loads(s) for s in manifest.read_text().splitlines()] if manifest.exists() else []
                if any(r['trajectory_id']==trajectory for r in existing):
                    raise ValueError('Concurrent duplicate trajectory capture')
                existing.extend(rows)
                temporary=manifest.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r)+'\n' for r in existing));temporary.replace(manifest)
                fcntl.flock(guard,fcntl.LOCK_UN)
            print(json.dumps(dict(trajectory=trajectory,split=split,inputs=len(rows),native_steps=steps)),flush=True)
    atomic_json(HERE/'reports/collection.json',dict(source_sha256=source_identity(),inputs=len(existing),
        parameterization='x0',native_steps=steps,teacher=p['teacher_sha256']))


def main():
    a=argparse.ArgumentParser();a.add_argument('--inventory',action='store_true')
    a.add_argument('--registry',action='store_true');a.add_argument('--split',choices=['train','val','test'])
    a.add_argument('--limit',type=int);a.add_argument('--offset',type=int,default=0)
    a.add_argument('--seeds',type=int,nargs='+',default=[0,1]);a.add_argument('--device',default='cuda')
    args=a.parse_args();paired_seed(0)
    if args.inventory:print(json.dumps(inventory(),ensure_ascii=False,indent=2))
    if args.registry:print(json.dumps({k:len(v) for k,v in prepare_registry()['splits'].items()}))
    if args.split:collect(args.split,args.limit,args.seeds,args.device,args.offset)


if __name__=='__main__':main()
