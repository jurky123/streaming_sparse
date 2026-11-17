"""Measure route+tail batches; keep total work at 72 query tiles/head."""
import json,gc
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .pipeline_graph import BatchGraphs
from .runtime import risk_from_logits
from .benchmark import measure
from ..review_v1.ops import Config

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ref,base,identity=inputs(True,True);rows=[]
    for t in (6,12,24,36,72):
        e=dict(base);e['q']=base['q'][:,:1].expand(-1,t,-1,-1).contiguous();e['own']=base['own'][:,:1].expand(-1,t).contiguous()
        rt=BatchGraphs(risk_from_logits,ref.phi,Config(),compile_route=True,compile_tail=True,action_kernel=True,tensor_math=True)
        def fn():
            out=rt.compute(e,True)
            return rt.compensate(out[1],e['s'],e['basis'],out[2],e['haar_float'])
        g,out=capture(fn);timing=measure(g.replay,15);rows.append(dict(tiles=t,ms=timing['gpu_ms_median'],layer_route_tail_ms=timing['gpu_ms_median']*72/t,allocated_bytes=torch.cuda.memory_allocated()))
        print(rows[-1],flush=True);del g,out,rt;gc.collect();torch.cuda.empty_cache()
    path=Path('artifacts/video_runtime_v1/hotspot_round5/tile_batch.json');path.write_text(json.dumps(dict(rows=rows,identity=identity),indent=2))
if __name__=='__main__':run()
