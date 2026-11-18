"""Verify the normal release CLI against the accepted experimental trajectory."""
import json
from pathlib import Path
import torch

root=Path(__file__).parent
dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True).float()
accepted=torch.load('/tmp/jzk_direct_mass_pair.pt',weights_only=True)
modes={}
for name in ('base','intersection','step'):
    folder=Path('/tmp/jzk_vector_deployed_'+name)
    run=json.loads((folder/'result.json').read_text())
    z=torch.load(folder/'latents.pt',weights_only=True)
    previous=torch.load(Path('/tmp/jzk_lanefma_deployed_'+name)/'latents.pt',weights_only=True)
    reference=accepted[1] if name=='base' else previous
    assert torch.equal(z.view(torch.uint8),reference.view(torch.uint8))
    stats=run['runtime_stats']
    assert stats['packed_attention_calls']==(812 if name=='base' else 0)
    modes[name]=dict(seconds=run['seconds'],median_seconds=run['median_seconds'],
                     cuda_calls=stats['packed_attention_calls'],commits=stats['direct_layout_commits'],
                     accepted_latent_bitwise=True,dense_nmse=float((z.float()-dense).square().sum()/dense.square().sum()),
                     previous_nmse=float((z.float()-previous.float()).square().sum()/previous.float().square().sum()))
(root/'vector_release.json').write_text(json.dumps(modes,indent=2)+'\n')
