"""Fresh evaluation of flattened compensation on accepted prefix-quality cases."""
import gc,json,os,signal,torch
from pathlib import Path
from .broad_common import Engine,cases,ROOT as DATA
from .gap_pool_strict_runtime import GapPoolStrictRuntime as GapFlatRuntime
from .rollout_suite import generate_chunk,video
from .high_sparse_experiment import prefix_decode,prefix_quality
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/gap_pool_20261031/strict_quality');OLD=Path('artifacts/four_gaps_20261031');MODEL=OLD/'shared/epoch1.pt'
@torch.no_grad()
def main():
    lock=RunLock(ROOT);stop=[False];rt=None
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        samples=[c|dict(seed=seed,id=f'test_{c["name"]}_s{seed}') for seed in [404,505] for c in cases('test') if c['seed']==202]
        samples=[c for c in samples if (c['name'],c['seed']) in [('bird_flight',505),('traffic_crossing',404)]]
        ident=dict(model=digest(MODEL),samples=samples,window=[21,101],screen=dict(mean_lpips_delta_max=.005,mean_ssim_delta_min=-.005,case_lpips_delta_max=.02,case_ssim_delta_min=-.02),scope='Two-case quick regression screen requested by user; not full quality acceptance or statistical noninferiority. Compare both against same dense early RGB.',code={n:digest(Path('local_experiments')/n) for n in ['gap_pool_strict_quality.py','gap_pool_strict_runtime.py','gap_cached_ops.py','gap_model.py']})
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==ident
        else:atomic(ROOT/'identity.json',ident)
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        rt=GapFlatRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=False)['model'])
        for case in samples:
            folder=ROOT/case['id'];folder.mkdir(exist_ok=True);dest=folder/'result.json'
            if dest.exists():continue
            if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted'));return
            disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache()
            old=OLD/f'test_s{case["seed"]}'/case['id']
            def decode(path):
                z=torch.load(path,weights_only=True).cuda();return prefix_decode(vae,list(z.split(3,1)))
            dense=decode(old/'dense.latents.pt');base=decode(Path('artifacts/gap_vector_flat_20261031/quality')/case['id']/'flat.latents.pt')
            baseline=prefix_quality(base,dense,metric)
            emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
            rt.reset()
            try:
                atomic(ROOT/'status.json',dict(status='running',case=case['id']))
                zs=[generate_chunk(engine,emb,case['seed'],c) for c in range(9)];frames=prefix_decode(vae,zs)
                quality=prefix_quality(frames,dense,metric);drift=prefix_quality(frames,base,metric)
                atomic(folder/'flat.latents.pt',torch.cat(zs,1).cpu(),True);video(folder/'flat.mp4',frames)
                delta={k:quality[k]-baseline[k] for k in ['lpips','ssim','psnr']}
                atomic(dest,dict(case=case,baseline=baseline,flat=quality,delta=delta,drift_from_accepted=drift,runtime=rt.finish(),baseline_video=str(Path('artifacts/gap_vector_flat_20261031/quality')/case['id']/'flat.mp4')))
                print(json.dumps(dict(case=case['id'],delta=delta,drift_lpips=drift['lpips'])),flush=True)
                del zs,frames,dense,base
            finally:pass
        rows=[json.loads((ROOT/c['id']/'result.json').read_text()) for c in samples]
        mean={k:sum(r['delta'][k] for r in rows)/len(rows) for k in ['lpips','ssim','psnr']}
        passed=mean['lpips']<=.005 and mean['ssim']>=-.005 and all(r['delta']['lpips']<=.02 and r['delta']['ssim']>=-.02 for r in rows)
        atomic(ROOT/'summary.json',dict(cases=len(rows),mean_delta=mean,passed_screen=passed,rows=[dict(case=r['case']['id'],delta=r['delta'],drift_lpips=r['drift_from_accepted']['lpips']) for r in rows],scope=ident['scope']))
        atomic(ROOT/'status.json',dict(status='complete',passed_screen=passed))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
