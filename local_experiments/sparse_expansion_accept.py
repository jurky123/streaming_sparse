"""Frozen factorial candidates; early RGB first, complete-unit resume and bounded storage."""
import argparse,gc,json,os,signal
from pathlib import Path
import torch,numpy as np
from .broad_common import Engine
from .sparse_expansion_fast_runtime import ExpansionRuntime
from .conditional_v_runtime import ConditionalRuntime
from .high_sparse_experiment import prefix_decode,prefix_quality,OLD
from .rollout_suite import generate_chunk,video,timed_run
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/sparse_expansion_20261031');MODEL='artifacts/full_route_grad_20261031/epoch1.pt'
CONFIGS={'p4_s1':(4,1,'single'),'p8_s1':(8,1,'single'),'p16_s1':(16,1,'single'),'p8_s2':(8,2,'single'),'p8_s4':(8,4,'single'),'p16_s2':(16,2,'single'),'p8_pair4':(8,1,'paired4'),'p8_haar1':(8,1,'haar1')}
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['early','full'],default='early');ap.add_argument('--configs',default=','.join(CONFIGS));args=ap.parse_args()
    lock=RunLock(ROOT/'acceptance');stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        for f in ['checks.json','fast_checks.json','conditional_checks.json']:assert json.loads((ROOT/f).read_text())['passed']
        files=['sparse_expansion_accept.py','sparse_expansion_runtime.py','sparse_expansion_fast_runtime.py','sparse_refine_kernel.py','conditional_v_runtime.py','conditional_v_kernels.py'];ident=dict(code={p:digest(Path('local_experiments')/p) for p in files},model=digest(MODEL),configs=CONFIGS,seed=1001,window=[21,101],extra_budget=16*2**20)
        ident=json.loads(json.dumps(ident));ip=ROOT/'acceptance/identity.json'
        if ip.exists():assert json.loads(ip.read_text())==ident
        else:atomic(ip,ident)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        for case in ['basketball_court','butterfly_garden']:
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda();saved=torch.load(OLD/(case+'_s1001')/'dense_host.latents.pt',weights_only=True).cuda();ref=prefix_decode(vae,list(saved[:,:27].split(3,1)));del saved
            for label in args.configs.split(','):
                folder=ROOT/'acceptance'/args.phase/case;dest=folder/(label+'.json')
                if dest.exists():continue
                if stop[0]:atomic(ROOT/'acceptance/status.json',dict(status='interrupted',phase=args.phase));return
                disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache();m,s,comp=CONFIGS[label];rt=ExpansionRuntime(engine,MODEL,m,s) if comp=='single' else ConditionalRuntime(engine,MODEL,m,s,comp)
                try:
                    atomic(ROOT/'acceptance/status.json',dict(status='running',case=case,config=label,phase=args.phase,pid=os.getpid()));perf=None
                    if args.phase=='early':
                        latents=[generate_chunk(engine,emb,1001,c) for c in range(9)];frames=prefix_decode(vae,latents)
                    else:
                        for c in range(6):warm=generate_chunk(engine,emb,1001,c)
                        vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                        tele=Telemetry();tele.start()
                        try:latents,frames,perf=timed_run(engine,rt,vae,emb,1001)
                        finally:load=tele.finish()
                        perf['telemetry']=load
                    value=torch.cat(latents,1).cpu();assert torch.isfinite(value).all();q=prefix_quality(frames,ref,metric)
                    if label=='p4_s1':assert torch.equal(value[:,:27],torch.load(Path('artifacts/full_route_grad_20261031/acceptance/early')/case/'trained.latents.pt',weights_only=True)),'Frozen baseline changed'
                    if args.phase=='full':assert torch.equal(value[:,:27],torch.load(ROOT/'acceptance/early'/case/(label+'.latents.pt'),weights_only=True))
                    atomic(folder/(label+'.latents.pt'),value,True);video(folder/(label+'.mp4'),frames)
                    atomic(dest,dict(case=case,config=label,phase=args.phase,quality=q,performance=perf,runtime=rt.finish(),note='Probe weights frozen from M4; feature distribution shifts with M. Must retrain all sparse modules separately.'))
                    print(json.dumps(dict(case=case,config=label,quality={k:v for k,v in q.items() if k!='per_frame'})),flush=True);del latents,frames,value
                finally:rt.close();del rt
        atomic(ROOT/'acceptance/status.json',dict(status='complete_requested',phase=args.phase,configs=args.configs))
    except Exception as exc:atomic(ROOT/'acceptance/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
