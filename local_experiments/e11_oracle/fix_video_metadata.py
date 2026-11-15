"""Recompute per-latent-frame curves from saved tensors; no generation changes."""
import json,torch
from pathlib import Path
ROOT=Path('/tmp/e11_results')
for path in (ROOT/'video').glob('*/*/*/result.json'):
 r=json.loads(path.read_text());z=torch.load(path.parent/'latents.pt',weights_only=True).double();ref=torch.load(r['reference'],weights_only=True).double();axes=tuple(i for i in range(z.ndim) if i!=1);curve=((z-ref).square().sum(axes)/ref.square().sum(axes)).tolist();assert len(curve)==27
 r['latent_per_frame_nmse']=curve;r['latent_curve_frames']=27
 if r['method']!='O-OPT':r['solver_seconds_per_instance']=None
 path.write_text(json.dumps(r,indent=2))
print('metadata checked')
