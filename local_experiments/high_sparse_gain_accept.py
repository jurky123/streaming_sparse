"""Resumable matched closed-loop ablations; checkpoint selected on validation only."""
import argparse,gc,json,os,signal
from pathlib import Path
import numpy as np
import torch
from .broad_common import Engine
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_experiment import prefix_decode,prefix_quality,OLD,CASES
from .rollout_suite import generate_chunk,video,timed_run,decode
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/high_sparse_gain_20261031')
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['early','full'],default='early');ap.add_argument('--modes',default='0.0,0.25,0.5,1.0');ap.add_argument('--cases',default=','.join(CASES));args=ap.parse_args()
    selection=json.loads((ROOT/'selection.json').read_text());cp=ROOT/'model.pt'
    assert json.loads((ROOT/'checks.json').read_text())['passed']
    out=ROOT/'acceptance';lock=RunLock(out);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        files=['high_sparse_gain_accept.py','high_sparse_gain_runtime.py','high_sparse_gain.py','high_sparse_learn_kernels.py','high_sparse_learn.py','high_sparse_runtime.py','compensation_stable_kernels.py']
        identity=dict(checkpoint=digest(cp),validation_selected_strength=selection['strength'],code={p:digest(Path('local_experiments')/p) for p in files},seed=1001,window=[21,101])
        ip=out/'identity.json'
        if ip.exists():assert json.loads(ip.read_text())==identity
        else:atomic(ip,identity)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        for case in args.cases.split(','):
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            saved=torch.load(OLD/(case+'_s1001')/'dense_host.latents.pt',weights_only=True).cuda()
            ref=prefix_decode(vae,list(saved[:,:27].split(3,1)));del saved
            folder=out/args.phase/case;folder.mkdir(parents=True,exist_ok=True)
            for mode in args.modes.split(','):
                path=folder/(mode+'.json')
                if path.exists():continue
                if stop[0]:atomic(out/'status.json',dict(status='interrupted',phase=args.phase));return
                disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache()
                atomic(out/'status.json',dict(status='running',case=case,mode=mode,phase=args.phase,pid=os.getpid()))
                rt=GainRuntime(engine,cp,float(mode))
                try:
                    performance=None
                    if args.phase=='early':
                        rt.reset();latents=[generate_chunk(engine,emb,1001,c) for c in range(9)];frames=prefix_decode(vae,latents)
                    else:
                        for c in range(6):warm=generate_chunk(engine,emb,1001,c)
                        vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                        telemetry=Telemetry();telemetry.start()
                        try:latents,frames,performance=timed_run(engine,rt,vae,emb,1001)
                        finally:load=telemetry.finish()
                        performance['telemetry']=load
                        prefix=torch.load(out/'early'/case/(mode+'.latents.pt'),weights_only=True)
                        assert torch.equal(torch.cat(latents,1)[:,:27].cpu(),prefix),'Full/prefix trajectory differs'
                    q=prefix_quality(frames,ref,metric);value=torch.cat(latents,1).cpu()
                    assert torch.isfinite(value).all() and np.array_equal(frames[0],ref[0])
                    
                    if float(mode)==0.:
                        prior=torch.load(Path('artifacts/high_sparse_train_20261030/acceptance/early')/case/'baseline.latents.pt',weights_only=True)
                        assert torch.equal(value[:,:27],prior),'Zero-correction trajectory differs from prior matched baseline'
                    runtime=rt.finish();assert runtime['budget_trace'][338]['selected_pages']==41
                    atomic(folder/(mode+'.latents.pt'),value,True);video(folder/(mode+'.mp4'),frames)
                    result=dict(case=case,mode=mode,phase=args.phase,quality=q,runtime=runtime,performance=performance,checkpoint=str(cp))
                    atomic(path,result);print(json.dumps({k:v for k,v in result.items() if k not in ['runtime','performance','quality']}|dict(quality={k:v for k,v in q.items() if k!='per_frame'})),flush=True)
                    del frames,latents,value
                finally:rt.close();del rt
        atomic(out/'status.json',dict(status='completed_requested_units',phase=args.phase))
    except Exception as exc:atomic(out/'status.json',dict(status='failed',phase=args.phase,error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
