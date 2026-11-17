"""Native Dense self-attention budget on the same single short video."""
import json,time,statistics
from pathlib import Path
import torch
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from .weights import ROOT
from .runtime import Timings

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();engine.wrapper.requires_grad_(False)
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    def generate():
        engine.reset()
        for c in range(9):generate_chunk(engine,emb,701,c)
    generate();torch.cuda.synchronize();rows=[]
    for _ in range(2):
        start=time.perf_counter();generate();torch.cuda.synchronize();rows.append(time.perf_counter()-start)
    import wan.modules.causal_model as module
    original=module.attention;timing=Timings(True)
    def measured(*args,**kwargs):
        cur=engine.current
        kind='layer0' if cur['layer']==0 else ('replaceable' if cur['chunk']>=2 and cur['step']<4 else 'retained')
        with timing.stage(kind):return original(*args,**kwargs)
    module.attention=measured
    start=time.perf_counter();generate();torch.cuda.synchronize();wall=time.perf_counter()-start
    result=dict(seconds=rows,median_seconds=statistics.median(rows),profiled_wall_seconds=wall,attention=timing.finish(),definition='replaceable: layers1-29, chunk>=2, four noisy steps; retained: clean and first two chunks in layers1-29; layer0 separately; profile events add overhead')
    path=Path('artifacts/video_runtime_v1/hotspot_round6/native_profile_v2.json');path.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':run()
