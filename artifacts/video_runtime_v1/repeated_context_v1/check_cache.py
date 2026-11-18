"""Validate the installed kernel, layout checks and repeated/nonrepeated transitions."""
from pathlib import Path
import json
from types import SimpleNamespace
import torch
from local_experiments.video_runtime_v1.repeated_context import RepeatedContext

patch=RepeatedContext(SimpleNamespace(cross=[{}]))
rows=[]
for q,k,v in torch.load('/tmp/jzk_cross_three.pt',weights_only=True):
    q,k,v=[x.cuda() for x in (q,k,v)]
    ref=torch.nn.functional.scaled_dot_product_attention(*(x.transpose(1,2) for x in (q,k,v))).transpose(1,2)
    packed=patch.prepare(k,v)
    out=patch.apply(q,packed)
    different=int((out.view(torch.int16)!=ref.view(torch.int16)).sum())
    assert not different
    for index in (0,1):
        pair=[k.clone(),v.clone()]
        pair[index][0,64,0,0]+=1
        assert patch.prepare(*pair) is None
    assert patch.prepare(k[:,:256],v[:,:256]) is None
    patch.enabled=False
    assert patch.prepare(k,v) is None
    patch.enabled=True
    assert patch.prepare(k,v) is not None
    rows.append(dict(different=different,packed_bytes=sum(x.numel()*x.element_size() for x in packed),
                     changed_key_fallback=True,changed_value_fallback=True,short_input_fallback=True))
patch.close()
Path(__file__).with_name('cache.json').write_text(json.dumps(rows,indent=2))
