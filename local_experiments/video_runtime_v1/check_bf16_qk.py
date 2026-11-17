import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .tensor_bmm import tensor_bmm,bf16_inputs_bmm
from .benchmark import measure

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    _,e,identity=inputs(True,True);rows=[]
    for m in (16,64):
        q=e['q'].index_select(2,e['offsets']) if m==16 else e['q']
        q=q.flatten(1,2);k=e['s'].keys.float().flatten(1,2).transpose(1,2)
        assert torch.equal(q,q.bfloat16().float()) and torch.equal(k,k.bfloat16().float())
        go,x=capture(lambda:tensor_bmm(q,k));gn,y=capture(lambda:bf16_inputs_bmm(q,k));r=torch.bmm(q,k)
        rows.append(dict(queries_per_tile=m,max_abs=float((x-y).abs().max()),nmse=float((x-y).square().sum()/x.square().sum()),reference_nmse=float((r-y).square().sum()/r.square().sum()),before=measure(go.replay,25),after=measure(gn.replay,25)))
    Path('artifacts/video_runtime_v1/hotspot_round6/bf16_qk.json').write_text(json.dumps(dict(rows=rows,identity=identity),indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':run()
