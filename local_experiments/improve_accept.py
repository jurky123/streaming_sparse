"""Frozen fresh-seed 30s generation+VAE acceptance, resumable whole-video units."""
import argparse,gc,json,os,signal,subprocess,threading,time
from pathlib import Path
import numpy as np
import torch
from .broad_common import Engine
from .rollout_runtime import Runtime
from .rollout_dense_control import HostDense
from .rollout_suite import generate_chunk,timed_run,decode,quality,video
from .improve_runtime import FastRuntime
from .improve_scatter_runtime import ScatterRuntime
from .improve_learned_runtime import LearnedRuntime
from .improve_eval_checks import ConfiguredKernel
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/rollout_improve_20261030')
CASES=['basketball_court','butterfly_garden','harbor_cranes','lantern_festival'];SEEDS=[808,909]
class Telemetry:
    def __init__(self):self.samples=[];self.stop=threading.Event()
    def sample(self):
        try:
            out=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
            pids=[int(x) for x in out.splitlines() if x.strip().isdigit()]
            self.samples.append(dict(time=time.time(),foreign=[p for p in pids if p!=os.getpid()]))
        except Exception as exc:self.samples.append(dict(time=time.time(),error=repr(exc)))
    def start(self):
        self.sample()
        def loop():
            while not self.stop.wait(5):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def finish(self):
        self.stop.set();self.thread.join();self.sample()
        return dict(no_observed_foreign_processes=all(not x.get('foreign') and 'error' not in x for x in self.samples),samples=self.samples,note='5-second GPU process samples; brief intervening tasks may escape sampling.')
def make(engine,method,selection,bench):
    if method=='dense':return Runtime(engine,'dense')
    if method=='dense_host':return HostDense(engine)
    if method=='mass_var':return Runtime(engine,'mass_var')
    if method=='route_fast':return FastRuntime(engine)
    if method=='scatter_fast':return ScatterRuntime(engine)
    if method=='kernel_fast':
        rt=FastRuntime(engine);rt.kernel=ConfiguredKernel(rt.kernel,bench['choice']);return rt
    return LearnedRuntime(engine,'anchor' if method=='pilot_anchor' else selection['selected_method'],proxy=method=='pilot_swap')
@torch.no_grad()
def main(args):
    selection=json.loads((ROOT/'selection.json').read_text());gate=json.loads((ROOT/'deployment_gate.json').read_text());bench=json.loads((ROOT/'kernel_benchmark.json').read_text());assert gate['passed']
    methods=['dense','dense_host','mass_var']+gate['engineering_variants']+['pilot_anchor','pilot_candidate']
    if gate['proxy']:methods.append('pilot_swap')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
    import lpips
    from utils.wan_wrapper import WanVAEWrapper
    engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    sources=['improve_accept.py','improve_runtime.py','improve_scatter_runtime.py','improve_scatter_kernel.py','improve_learned_runtime.py','improve_learned_kernels.py','improve_models.py','improve_eval_checks.py','rollout_runtime.py','rollout_kernels.py','rollout_suite.py','rollout_dense_control.py','broad_common.py']
    identity=dict(code={s:digest(Path('local_experiments')/s) for s in sources},weights=digest(ROOT/'pilot/best.pt'),selection=selection,gate=gate,methods=methods,repeats=3,
        cases=CASES,seeds=SEEDS,protocol='Frozen 6-layer pilot; physical338+aux<=native360;72historical+72current;absoluteRoPE;30s480frames;wall generation+VAE;fresh seeds, existing prompts')
    units=0
    for case in CASES:
        for seed in SEEDS:
            root=ROOT/'acceptance'/f'{case}_s{seed}';lock=RunLock(root);reference=None
            try:
                ident=identity|dict(case=case,seed=seed)
                if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==ident,'Acceptance identity mismatch'
                else:atomic(root/'identity.json',ident)
                if all((root/f'{m}.rep{r}.json').exists() for m in methods for r in range(3)):continue
                emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
                for rep in range(3):
                    order=methods[rep*2:]+methods[:rep*2]
                    for method in order:
                        path=root/f'{method}.rep{rep}.json'
                        if path.exists():continue
                        if stop[0]:break
                        disk_guard(ROOT,50,4);gc.collect();torch.cuda.empty_cache();rt=make(engine,method,selection,bench)
                        try:
                            for ch in range(6):warm=generate_chunk(engine,emb,seed,ch)
                            vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                            telemetry=Telemetry();telemetry.start()
                            try:latents,frames,result=timed_run(engine,rt,vae,emb,seed)
                            finally:load=telemetry.finish()
                            result.update(method=method,telemetry=load,case=case,seed=seed,rep=rep,completed=True)
                            assert rt.commits[1:]==([41]*29 if rt.grouped else [0]*29)
                            value=torch.cat(latents,1).cpu();lpath=root/f'{method}.latents.pt'
                            if rep==0:
                                if method=='dense':reference=frames
                                if reference is None:
                                    original=torch.load(root/'dense.latents.pt',weights_only=True).cuda();reference=decode(vae,list(original.split(3,1)));del original
                                assert np.array_equal(frames[0],reference[0]),'first9 frames changed'
                                alias='dense' if method=='dense_host' else 'mass_var' if method in gate['engineering_variants'] else None
                                if alias:
                                    assert torch.equal(value,torch.load(root/f'{alias}.latents.pt',weights_only=True)),f'{method} changed fresh-seed video'
                                    # Still measure actual uncompressed RGB quality; alias only after latent equality.
                                qm=quality(frames,reference,metric);atomic(root/f'{method}.quality.json',qm)
                                if alias:
                                    if not lpath.exists():lpath.symlink_to(f'{alias}.latents.pt')
                                    vpath=root/f'{method}.mp4'
                                    if not vpath.exists():vpath.symlink_to(f'{alias}.mp4')
                                    result['video_alias_exact_latents']=alias
                                else:atomic(lpath,value,True);video(root/f'{method}.mp4',frames)
                                result['quality']={k:v for k,v in qm.items() if k!='per_frame'}
                            else:assert torch.equal(value,torch.load(lpath,weights_only=True)),f'{method} repeat not exact'
                            atomic(path,result);units+=1
                            atomic(root/'status.json',dict(status='running',method=method,rep=rep,disk=disk_guard(ROOT,50,4)))
                            print(json.dumps(dict(case=case,seed=seed,method=method,rep=rep,total_s=result['total_generation_vae_seconds'],clean=load['no_observed_foreign_processes'])),flush=True)
                            del latents,frames,value
                        finally:rt.close();del rt
                        if args.max_units and units>=args.max_units:stop[0]=True
                    if stop[0]:break
                done=all((root/f'{m}.rep{r}.json').exists() for m in methods for r in range(3))
                atomic(root/'status.json',dict(status='completed' if done else 'interrupted'))
            except Exception as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
            finally:lock.close()
            if stop[0]:return
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-units',type=int,default=0);main(p.parse_args())
