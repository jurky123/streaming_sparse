"""Deployment gate: exact engineering outputs and learned cache/score/kernel checks."""
import gc,json
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .improve_runtime import FastRuntime,AuditKernel
from .improve_scatter_runtime import ScatterRuntime
from .improve_learned_runtime import LearnedRuntime
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/rollout_improve_20261030')
class ConfiguredKernel:
    def __init__(self,kernel,choice):self.kernel=kernel;self.choice=choice
    def __getitem__(self,grid):
        def call(*args,**kw):
            kw.update(num_warps=self.choice[0],num_stages=self.choice[1]);return self.kernel[grid](*args,**kw)
        return call
@torch.no_grad()
def main():
    root=ROOT/'deployment_checks';lock=RunLock(root)
    try:
        selection=json.loads((ROOT/'selection.json').read_text());bench=json.loads((ROOT/'kernel_benchmark.json').read_text())
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity=dict(selection=digest(ROOT/'selection.json'),weights=digest(ROOT/'pilot/best.pt'),code={f:digest(Path('local_experiments')/f) for f in ['improve_eval_checks.py','improve_scatter_runtime.py','improve_scatter_kernel.py','improve_learned_runtime.py','improve_learned_kernels.py','improve_models.py']})
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        engine=Engine();todo=[('scatter_fast',c) for c in ['basketball_court','butterfly_garden']]
        if bench['promoted']:todo += [('kernel_fast',c) for c in ['basketball_court','butterfly_garden']]
        todo += [('pilot_anchor','horse_field'),('pilot_candidate','horse_field')]
        if selection['proxy_promoted']:todo.append(('pilot_swap','horse_field'))
        for method,case in todo:
            path=root/f'{case}.{method}.json'
            if path.exists():continue
            folder='broad_coverage_20261030' if case=='horse_field' else 'mass_variance_20261030'
            emb=torch.load(f'artifacts/{folder}/embeddings/{case}.pt',weights_only=True).cuda();seed=101 if case=='horse_field' else 606
            if method=='scatter_fast':rt=ScatterRuntime(engine)
            elif method=='kernel_fast':
                rt=FastRuntime(engine);rt.kernel=ConfiguredKernel(rt.kernel,bench['choice'])
            else:rt=LearnedRuntime(engine,'anchor' if method=='pilot_anchor' else selection['selected_method'],proxy=method=='pilot_swap',audit_scores=True)
            auditor=AuditKernel(rt,rt.kernel);rt.kernel=auditor;latents=[];center_error=0.
            try:
                for ch in range(41):
                    latents.append(generate_chunk(engine,emb,seed,ch).cpu())
                    if isinstance(rt,LearnedRuntime) and ch in [5,10,20,30,40]:
                        for l,model in rt.models.items():
                            k=engine.cache[l]['k'][0,:338*64].transpose(0,1).float().contiguous();pk,_,count=model(dict(k=k))
                            torch.testing.assert_close(rt.pk[l],pk,atol=5e-5,rtol=5e-4)
                            if l in rt.centers:
                                error=(rt.centers[l]-k.mean(1)).abs().max().item();center_error=max(center_error,error)
                                torch.testing.assert_close(rt.centers[l],k.mean(1),atol=3e-6,rtol=2e-4)
                            if l in rt.logcounts:torch.testing.assert_close(rt.logcounts[l],count.log()-__import__('math').log(16),atol=3e-5,rtol=3e-4)
                    disk_guard(ROOT,50,4)
                out=torch.cat(latents,1);exact=None
                if method in ['scatter_fast','kernel_fast']:
                    reference=torch.load(f'artifacts/rollout_acceptance_20261030/{case}/mass_var.latents.pt',weights_only=True);exact=torch.equal(out,reference)
                else:
                    reference=torch.load('artifacts/broad_coverage_20261030/trajectories/val_horse_field_s101/chunk00.pt',weights_only=True)['clean']
                    assert torch.equal(out[:,:3],reference)
                atomic(path,dict(completed=True,method=method,case=case,exact_primary_latents=exact,kernel_checks=auditor.rows,max_center_error=center_error,runtime=rt.finish()))
                print(case,method,'passed; exact=',exact,flush=True)
            finally:rt.close();del rt,latents;gc.collect();torch.cuda.empty_cache()
        promoted=['route_fast']
        for m in ['scatter_fast','kernel_fast']:
            paths=[root/f'{c}.{m}.json' for c in ['basketball_court','butterfly_garden']]
            if all(p.exists() and json.loads(p.read_text())['exact_primary_latents'] for p in paths):promoted.append(m)
        atomic(ROOT/'deployment_gate.json',dict(passed=True,engineering_variants=promoted,learned_layers=selection['layers'],candidate=selection['selected_method'],proxy=selection['proxy_promoted'],
            rejected_commit_fusion='Original arithmetic-changing commit_fast did not preserve full-video latents.'))
        atomic(root/'status.json',dict(completed=True))
    except Exception as exc:atomic(root/'status.json',dict(completed=False,error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
