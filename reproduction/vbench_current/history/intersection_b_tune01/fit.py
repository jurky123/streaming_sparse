from pathlib import Path
import json
import torch
from local_experiments.video_runtime_v1.deployment import CURRENT
out=Path(__file__).parent
train=('forest_run','reef_fish','toy_train','fabric_dance')
val=('kitchen_cook','snow_ski')
def load(name):
    return torch.load(out/(name+'.pt'),weights_only=True)['stats']
def unpack(s):
    return s[...,:64].unflatten(-1,(8,8)),s[...,64:72]
s=sum(load(n) for n in train).sum(2)
gram,rhs=unpack(s)
ridge=.2*gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
delta=torch.linalg.solve(gram+ridge[...,None,None]*torch.eye(8),rhs.unsqueeze(-1)).squeeze(-1).clamp(-.36,.44)
gain=(1+delta).sqrt()
def metrics(name):
    st=load(name)
    g,r=unpack(st)
    change=torch.einsum('lhk,lhtkj,lhj->lht',delta,g,delta)-2*torch.einsum('lhk,lhtk->lht',delta,r)
    base=st[...,72].sum((0,1))
    return dict(before=(base/st[...,73].sum((0,1))).tolist(),projected=((base+change.sum((0,1)))/st[...,73].sum((0,1))).tolist(),
                relative_change=float(change.sum()/base.sum()))
record=torch.load(CURRENT['checkpoint'],map_location='cpu',weights_only=False)
state=record['model']
for l in range(1,30):
    b=state[f'layers.{l}.basis']
    b.mul_(gain[l-1,:,None,:].to(b.dtype))
res=dict(train=list(train),val=list(val),ridge_fraction=.2,basis_gain_range=[float(gain.min()),float(gain.max())],
         gain_mean=float(gain.mean()),gain_std=float(gain.std()),metrics={n:metrics(n) for n in (*train,*val)})
torch.save(dict(model=state,adaptation=res),out/'candidate.pt')
(out/'fit.json').write_text(json.dumps(res,indent=2))
