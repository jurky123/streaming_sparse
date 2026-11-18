"""Track the first changed latent chunk and compare both runs to the same Dense output."""
from pathlib import Path
import json
import torch

pair=torch.load('/tmp/jzk_direct_mass_pair.pt',weights_only=True)
dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True)
def compare(old,new):
    delta=new.float()-old.float()
    return dict(changed=int((new!=old).sum()),nmse=float(delta.square().sum()/old.float().square().sum()),
                max_abs=float(delta.abs().max()))
rows=[]
for a,b,ref in zip(pair[0].chunk(9,dim=1),pair[1].chunk(9,dim=1),dense.chunk(9,dim=1)):
    rows.append(dict(pair=compare(a,b),base_dense=compare(ref,a),candidate_dense=compare(ref,b)))
(Path(__file__).parent/'chunk_quality.json').write_text(json.dumps(rows,indent=2))
print([(r['pair']['changed'],round(r['pair']['nmse'],5)) for r in rows])
