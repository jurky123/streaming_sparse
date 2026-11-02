"""Accepted shared checkpoint: paired full-video timing and exact latent audit."""
import gc,json,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_fast_runtime import GapFastRuntime
from .gap_feature_runtime import GapFeatureRuntime
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk,timed_run
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/gap_profile_20261031/performance_features')
MODEL=Path('artifacts/four_gaps_20261031/shared/epoch1.pt')
@torch.no_grad()
def main():
    lock=RunLock(ROOT)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from utils.wan_wrapper import WanVAEWrapper
        identity=dict(model=digest(MODEL),case='horse_field',seed=101,code={n:digest(Path('local_experiments')/n) for n in ['gap_profile_feature_perf.py','gap_feature_runtime.py','gap_feature_kernel.py','gap_repstats.py']})
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for unit in ['fast_0','fused_0','fused_1','fast_1']:
            dest=ROOT/(unit+'.json')
            if dest.exists():continue
            disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache();name=unit.split('_')[0]
            rt=Runtime(engine,'dense') if name=='dense' else {'fast':GapFastRuntime,'fused':GapFeatureRuntime}[name](engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
            if name!='dense':rt.model.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=False)['model'])
            try:
                atomic(ROOT/'status.json',dict(status='running',unit=unit))
                for c in range(6):warm=generate_chunk(engine,emb,101,c)
                vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                zs,frames,perf=timed_run(engine,rt,vae,emb,101);value=torch.cat(zs,1).cpu();exact=None
                if name!='dense':
                    path=ROOT/'reference.latents.pt'
                    if path.exists():exact=torch.equal(value,torch.load(path,weights_only=True))
                    else:atomic(path,value,True);exact=True
                    assert exact,'Fused full rollout differs'
                result=dict(unit=unit,performance=perf,full_latents_exact=exact,model=str(MODEL) if name!='dense' else 'dense')
                atomic(dest,result);print(json.dumps(dict(unit=unit,seconds=perf['total_generation_vae_seconds'],exact=exact)),flush=True)
                del zs,frames,value
            finally:rt.close();del rt
        atomic(ROOT/'status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
