"""Full generation+VAE, warmed paired execution, actual latent equality."""
import gc,json,time,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_runtime import GapRuntime
from .gap_graph_runtime import GapGraphRuntime
from .gap_fast_runtime import GapFastRuntime
from .sparse_expansion_fast_runtime import ExpansionRuntime
from .rollout_suite import generate_chunk,timed_run,video
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/four_gaps_20261031');BASE=Path('artifacts/full_route_grad_20261031/epoch1.pt')
@torch.no_grad()
def main():
    root=ROOT/'performance';lock=RunLock(root)
    try:
        for path in [ROOT/'fast_checks.json',ROOT/'cost.json',ROOT/'selection.json']:
            while not path.exists():time.sleep(10)
        assert json.loads((ROOT/'fast_checks.json').read_text())['passed']
        choice=json.loads((ROOT/'selection.json').read_text())['selected']['selected_matched'];model_path=Path(choice['path'])
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity=dict(model=digest(model_path),selection=choice,case='horse_field',seed=101,order=['reference_0','graph_0','fast_0','old16_0','old16_1','fast_1','graph_1','reference_1'],code={n:digest(Path('local_experiments')/n) for n in ['gap_full_perf.py','gap_runtime.py','gap_graph_runtime.py','gap_fast_runtime.py','gap_cached_ops.py']})
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for unit in identity['order']:
            dest=root/(unit+'.json')
            if dest.exists():continue
            disk_guard(root,50,1);gc.collect();torch.cuda.empty_cache();name=unit.rsplit('_',1)[0];cls={'reference':GapRuntime,'graph':GapGraphRuntime,'fast':GapFastRuntime}.get(name)
            rt=ExpansionRuntime(engine,BASE,16,1) if name=='old16' else cls(engine,BASE,probes=16)
            if name!='old16':rt.model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=False)['model'])
            try:
                atomic(root/'status.json',dict(status='running',unit=unit))
                for c in range(6):warm=generate_chunk(engine,emb,101,c)
                vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                tele=Telemetry();tele.start()
                try:zs,frames,perf=timed_run(engine,rt,vae,emb,101)
                finally:load=tele.finish()
                value=torch.cat(zs,1).cpu();early=ROOT/'validation/val_horse_field_s101'/((choice['label'] if name!='old16' else 'old16')+'.latents.pt')
                assert torch.equal(value[:,:27],torch.load(early,weights_only=True)),'Full/prefix differs'
                refpath=root/('old16.latents.pt' if name=='old16' else 'reference.latents.pt')
                if refpath.exists():assert torch.equal(value,torch.load(refpath,weights_only=True)),'Optimized full rollout differs'
                else:atomic(refpath,value,True);video(root/('old16.mp4' if name=='old16' else 'new.mp4'),frames)
                atomic(dest,dict(performance=perf,telemetry=load,runtime=rt.finish(),full_latents_exact=True,model=str(model_path) if name!='old16' else str(BASE)));print(json.dumps(dict(unit=unit,seconds=perf['total_generation_vae_seconds'],generation=perf['generation_seconds'])),flush=True);del zs,frames,value
            finally:rt.close();del rt
        atomic(root/'status.json',dict(status='complete'))
    except Exception as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
