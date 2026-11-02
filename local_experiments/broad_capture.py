import argparse
import gc
import json
import signal
from pathlib import Path
import time
import torch
from .broad_common import ROOT,CHUNKS,cases,guard,manifest,Engine,noise
from .learned_cache.io import atomic,RunLock,digest


@torch.no_grad()
def encode():
    from utils.wan_wrapper import WanTextEncoder
    missing=[c for c in cases() if not (ROOT/'embeddings'/f'{c["name"]}.pt').exists()]
    if not missing:return
    encoder=WanTextEncoder().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    for c in missing:
        dest=ROOT/'embeddings'/f'{c["name"]}.pt'
        if dest.exists():continue
        guard();atomic(dest,encoder([c['prompt']])['prompt_embeds'].cpu(),True);print('encoded',c['name'],flush=True)
    del encoder;gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def capture(args):
    manifest();stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    if args.encode:encode();return
    engine=Engine()
    for index,case in enumerate(cases()):
        if index%args.workers!=args.worker:continue
        folder=ROOT/'trajectories'/case['id'];lock=RunLock(folder)
        try:
            if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text())['status']=='completed':continue
            identity=dict(case=case,checkpoint=json.loads((ROOT/'manifest.json').read_text())['checkpoint'],
                common=digest('local_experiments/broad_common.py'),capture=digest(__file__),embedding=digest(ROOT/'embeddings'/f'{case["name"]}.pt'))
            if (folder/'identity.json').exists():assert json.loads((folder/'identity.json').read_text())==identity
            else:atomic(folder/'identity.json',identity)
            engine.reset();engine.callback=None;emb=torch.load(ROOT/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda();tick=time.monotonic()
            for chunk in range(41):
                path=folder/f'chunk{chunk:02d}.pt'
                if path.exists():
                    old=torch.load(path,weights_only=True);engine.forward(old['clean'].cuda(),emb,chunk,4);continue
                if stop[0]:break
                guard();samples={};noisy=[]
                def callback(current,q,k,v):
                    if current['step']==4:return
                    count=16 if case['split']=='train' else 4
                    ids=torch.linspace(0,q.shape[1]-1,count,device='cuda').long()
                    samples[(current['step'],current['layer'])]=q[0,ids].transpose(0,1).contiguous().cpu()
                engine.callback=callback if chunk in CHUNKS else None
                x=noise(case['seed'],chunk,0)
                for step in range(4):
                    if chunk in CHUNKS:noisy.append(x.cpu())
                    clean=engine.forward(x,emb,chunk,step)
                    if step<3:
                        x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),noise(case['seed'],chunk,step+1).flatten(0,1),
                            torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
                engine.callback=None;engine.forward(clean,emb,chunk,4)
                record=dict(clean=clean.cpu(),chunk=chunk)
                if noisy:record.update(noisy=torch.stack(noisy),queries=torch.stack([torch.stack([samples[(s,l)] for l in range(30)]) for s in range(4)]))
                atomic(path,record,True);atomic(folder/'status.json',dict(status='running',chunks=chunk+1,seconds=time.monotonic()-tick))
                print(json.dumps(dict(case=case['id'],chunk=chunk+1,seconds=round(time.monotonic()-tick,2))),flush=True)
                if args.max_new_chunks:
                    args.max_new_chunks-=1
                    if args.max_new_chunks==0:stop[0]=True
            completed=all((folder/f'chunk{c:02d}.pt').exists() for c in range(41))
            atomic(folder/'status.json',dict(status='completed' if completed else 'interrupted',chunks=sum((folder/f'chunk{c:02d}.pt').exists() for c in range(41)),disk=guard()))
        finally:lock.close()
        if stop[0]:break


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encode',action='store_true');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--max-new-chunks',type=int,default=0)
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;capture(p.parse_args())
