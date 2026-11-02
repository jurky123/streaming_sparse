"""Full 30s regression, late kernel oracle, ring alignment and exact-mask fast paths."""
import gc,json
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk
from .rollout_kernels import aggregate
from .improve_runtime import FastRuntime,AuditKernel
from .learned_cache.io import atomic,RunLock,digest,disk_guard
ROOT=Path('artifacts/rollout_improve_20261030')

@torch.no_grad()
def main():
    root=ROOT/'correctness';lock=RunLock(root)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity={f:digest(Path('local_experiments')/f) for f in ['improve_audit.py','improve_runtime.py','improve_kernels.py','rollout_runtime.py','rollout_kernels.py']}
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        engine=Engine()
        for case in ['butterfly_garden','basketball_court']:
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            original=torch.load(f'artifacts/rollout_acceptance_20261030/{case}/mass_var.latents.pt',weights_only=True)
            for variant in ['baseline_audit','route_fast','commit_fast']:
                path=root/f'{case}.{variant}.json'
                if path.exists():continue
                disk_guard(ROOT,min_free_gib=50,max_run_gib=4)
                rt=Runtime(engine,'mass_var') if variant=='baseline_audit' else FastRuntime(engine,variant,check_routes=True)
                auditor=AuditKernel(rt,rt.kernel);rt.kernel=auditor;latents=[];max_proto_error=0.
                try:
                    for ch in range(41):
                        latents.append(generate_chunk(engine,emb,606,ch).cpu())
                        if ch in [5,10,20,30,40]:
                            for l in [1,14,29]:
                                k=engine.cache[l]['k'][0,:338*64].transpose(0,1).contiguous().float()
                                target=aggregate(k,rt.weights[l]);delta=(target-rt.pk[l]).abs().max().item();max_proto_error=max(max_proto_error,delta)
                                torch.testing.assert_close(rt.pk[l],target,atol=3e-5,rtol=3e-4)
                        print(json.dumps(dict(case=case,variant=variant,chunk=ch)),flush=True)
                    output=torch.cat(latents,1);exact=torch.equal(output,original)
                    # Baseline must reproduce frozen run. Engineering variants are reported, not forced to pass.
                    if variant=='baseline_audit':assert exact,'baseline reproduction failed'
                    atomic(path,dict(completed=True,case=case,variant=variant,exact_primary_latents=exact,
                        latent_nmse=float((output.float()-original.float()).square().sum()/original.float().square().sum()),
                        max_prototype_error=max_proto_error,kernel_checks=auditor.rows,runtime=rt.finish()))
                finally:rt.close();del rt,latents;gc.collect();torch.cuda.empty_cache()
        atomic(root/'status.json',dict(completed=True))
    except Exception as exc:atomic(root/'status.json',dict(completed=False,error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
