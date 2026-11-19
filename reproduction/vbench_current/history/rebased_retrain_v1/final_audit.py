from pathlib import Path
import json,hashlib
import torch
root=Path(__file__).parent
repo=Path('/home/shuchenweng/jzk/streaming_sparse')
release=repo/'artifacts/full_coverage_v1/training'
def ck(p):return torch.load(p,weights_only=False,map_location='cpu')['model']
old_c=ck(release/'C_s17/deployment.pt');old_r=ck(release/'Full_Pset_s17/selected.pt')
def rms_change(a,b):
    return (sum((a[k].double()-b[k].double()).square().sum().item() for k in a)/sum(t.numel() for t in a.values()))**.5
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={}
for idx in (11,21):
    d=root/f'budget_{idx}';new_c=ck(d/'summary.pt');new_r=ck(d/'reader.pt')
    assert old_c.keys()==new_c.keys() and old_r.keys()==new_r.keys()
    touched=[k for k in old_c if not torch.equal(old_c[k],new_c[k])]
    assert all(k.rsplit('.',1)[-1] in ('pool_k','pool_v','pool_feature') for k in touched)
    assert len(touched)==29*3
    assert all(torch.isfinite(v).all() for v in list(new_c.values())+list(new_r.values()))
    out[idx]=dict(C_changed_tensors=len(touched),C_other_tensors_bit_exact=len(old_c)-len(touched),C_rms_all=rms_change(old_c,new_c),reader_rms=rms_change(old_r,new_r),summary_sha256=sha(d/'summary.pt'),reader_sha256=sha(d/'reader.pt'))
for p,h in json.loads((root/'identity.json').read_text())['frozen_sources'].items():
    assert sha(repo/p)==h
(root/'weights_audit.json').write_text(json.dumps(out,indent=2))
