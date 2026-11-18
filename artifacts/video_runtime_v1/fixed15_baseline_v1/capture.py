"""Save the first real instance at each window, then replay the accepted 30s fixture."""
import sys
import torch
from pathlib import Path
from local_experiments.video_runtime_v1 import legacy_components,preview_sink_chunk as preview
original=legacy_components.attention
seen=set()
def capture(q,k,v,s,basis,ids,workspace,**options):
    p=k.shape[1]//64
    if p not in seen:
        seen.add(p)
        assert ids.shape[-1]==p*15//100
        torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),ids=ids.cpu(),basis=basis.cpu(),
                        hp=s.mean.stride(0)//128,
                        summary={n:getattr(s,n).cpu() for n in ('keys','mean','detail')}),
                   f'/tmp/jzk_fixed15_input_{p}.pt')
    return original(q,k,v,s,basis,ids,workspace,**options)
legacy_components.attention=capture
preview.install=lambda rt,*args:rt.set_first_frame_fixed15(True)
sys.argv=['preview','/tmp/forcingkv-preview.LGIEZM/input.pt','/tmp/jzk_fixed15_verify.mp4','--mode','anchor-fixed','--fixed-percent','15']
preview.main()
