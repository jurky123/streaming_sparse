"""Resumable full-video numerical/quality audit followed by paired speed trials."""
import argparse,gc,json,os,signal
from pathlib import Path
import numpy as np
import torch
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .broad_common import Engine
from .rollout_suite import generate_chunk,timed_run,decode,quality,video
from .compensation_optimized_runtime import OptimizedCompensationRuntime
from .improve_runtime import FastRuntime
from .rollout_dense_control import HostDense
from .improve_accept import Telemetry
ROOT=Path('artifacts/compensation_optimized_accept_20261030')
OLD=Path('artifacts/compensation_probe_20261030/acceptance')
CONFIG=(64,16,4,1,False)
def early(q):
    p=q['per_frame'];m=float(np.mean(p['mse'][21:101]))
    return dict(lpips=float(np.mean(p['lpips'][21:101])),ssim=float(np.mean(p['ssim'][21:101])),psnr=100. if m==0 else float(-10*np.log10(m)))
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        ident=dict(code={s:digest(Path('local_experiments')/s) for s in ['compensation_optimized_accept.py','compensation_optimized_runtime.py','compensation_split_kernels.py','compensation_fast_kernels.py','compensation_kernels.py','compensation_runtime.py']},model=digest('artifacts/compensation_probe_20261030/model.pt'),config=CONFIG,backend='split',seed=1001,quality_window=[21,101],gate_tolerances=dict(lpips=.005,ssim=.005,psnr=.2))
        ident=json.loads(json.dumps(ident))
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==ident
        else:atomic(ROOT/'identity.json',ident)
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        for case in ['basketball_court','butterfly_garden']:
            folder=ROOT/case;folder.mkdir(exist_ok=True);old=OLD/(case+'_s1001')
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            gatepath=folder/'gate.json'
            if not gatepath.exists():
                if stop[0]:return
                disk_guard(ROOT,50,1)
                rt=OptimizedCompensationRuntime(engine,'split',CONFIG,audit=True)
                try:
                    latents,frames,result=timed_run(engine,rt,vae,emb,1001)
                    assert len(rt.comp_checks)==232,len(rt.comp_checks)
                    value=torch.cat(latents,1).cpu();assert torch.isfinite(value).all()
                    saved=torch.load(old/'dense_host.latents.pt',weights_only=True).cuda();ref=decode(vae,list(saved.split(3,1)));del saved
                    assert np.array_equal(frames[0],ref[0])
                    qm=quality(frames,ref,metric);atomic(folder/'split.quality.json',qm)
                    original=early(json.loads((old/'rank8_raw.quality.json').read_text()));current=early(qm)
                    passed=current['lpips']<=original['lpips']+.005 and current['ssim']>=original['ssim']-.005 and current['psnr']>=original['psnr']-.2
                    raw=torch.load(old/'rank8_raw.latents.pt',weights_only=True)
                    gate=dict(passed=passed,current=current,original=original,checks=rt.comp_checks,full_latents_exact_to_raw=torch.equal(value,raw),latent_nmse=float((value.float()-raw.float()).square().sum()/raw.float().square().sum()),persistent_bytes=rt.persistent_bytes,note='Audit timing excluded from speed acceptance; existing scenes regression, not heldout quality evidence.')
                    atomic(folder/'split.latents.pt',value,True);video(folder/'split.mp4',frames);atomic(gatepath,gate)
                    print(json.dumps(dict(case=case,gate={k:v for k,v in gate.items() if k!='checks'})),flush=True)
                    del latents,frames,value,ref,raw
                finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
            assert json.loads(gatepath.read_text())['passed'],f'Quality regression: {case}'
        if args.audit_only:return
        for rep in range(args.repeats):
            for case in ['basketball_court','butterfly_garden']:
                folder=ROOT/case;emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
                methods=['dense_host','mass_var','split'];methods=methods[rep%3:]+methods[:rep%3]
                for method in methods:
                    path=folder/f'{method}.rep{rep}.json'
                    if path.exists():continue
                    if stop[0]:return
                    disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache()
                    rt=HostDense(engine) if method=='dense_host' else FastRuntime(engine) if method=='mass_var' else OptimizedCompensationRuntime(engine,'split',CONFIG)
                    try:
                        for c in range(6):warm=generate_chunk(engine,emb,1001,c)
                        vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                        telemetry=Telemetry();telemetry.start()
                        try:latents,frames,result=timed_run(engine,rt,vae,emb,1001)
                        finally:load=telemetry.finish()
                        reference=folder/'split.latents.pt' if method=='split' else OLD/(case+'_s1001')/(method+'.latents.pt')
                        assert torch.equal(torch.cat(latents,1).cpu(),torch.load(reference,weights_only=True)),'Closed-loop repeat differs'
                        result.update(case=case,method=method,rep=rep,telemetry=load);atomic(path,result)
                        print(json.dumps(dict(case=case,method=method,rep=rep,seconds=result['total_generation_vae_seconds'],clean=load['no_observed_foreign_processes'])),flush=True)
                        del latents,frames
                    finally:rt.close();del rt
        atomic(ROOT/'status.json',dict(status='completed',repeats=args.repeats))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted'))
        lock.close()
if __name__=='__main__':main()
