"""Post-hoc diagnostic references, explicitly separate from predeclared acceptance."""
import json,os
from pathlib import Path
import torch
from .rollout_runtime import ROOT,METHODS
from .rollout_suite import decode,quality,guard
from .rollout_analyze import CASES
from .learned_cache.io import atomic,RunLock,digest

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['TORCH_HOME']=str((ROOT/'torch_cache').resolve())
    import lpips
    from utils.wan_wrapper import WanVAEWrapper
    root=ROOT/'secondary_quality';lock=RunLock(root)
    try:
        vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        for case in CASES:
            if not all((ROOT/case/f'{m}.rep0.json').exists() for m in METHODS):continue
            def pixels(m):
                z=torch.load(ROOT/case/f'{m}.latents.pt',weights_only=True).cuda()
                return decode(vae,list(z.split(3,1)))
            refs={m:pixels(m) for m in ['dense_trim','dense_grouped']}
            for m in ['mean','mass_mean','mass_var']:
                if all((root/f'{case}.{m}.vs_{ref}.json').exists() for ref in refs):continue
                frames=pixels(m)
                for ref,reference in refs.items():
                    path=root/f'{case}.{m}.vs_{ref}.json'
                    if path.exists():continue
                    guard();q=quality(frames,reference,metric)
                    q.update(case=case,method=m,reference=ref,post_hoc=True,note='Diagnostic only; does not replace native-dense acceptance reference or thresholds. Closed-loop divergence is not an isolated attention error.',source_latents={x:digest(ROOT/case/f'{x}.latents.pt') for x in [m,ref]})
                    atomic(path,q);print(json.dumps({k:v for k,v in q.items() if k!='per_frame'}),flush=True)
                del frames
            del refs
    finally:lock.close()

if __name__=='__main__':main()
