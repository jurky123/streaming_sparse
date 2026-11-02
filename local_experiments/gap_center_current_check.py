"""Real trained model/cache summary layout audit and local graph benchmark."""
import torch,triton,json
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_ordered_runtime import GapFlatOrderedRuntime as GapGroupRuntime
from .gap_pool_exact_runtime import GapPoolExactRuntime as GapSummaryReuseRuntime
from .gap_center_runtime import GapCenterRuntime as GapLayoutRuntime
from .gap_center_ops import centered_pool
from .gap_pool_ops import fused_pool
from .gap_cached_ops import cached_pool
import types
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();rt=GapGroupRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
    rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
    emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda();rows=[]
    try:
        for c in range(6):generate_chunk(engine,emb,101,c)
        for l in [1,14,29]:
            target=[v.clone() for v in rt.current_summary(l)]
            graphs={name:{} for name in ['native','reuse','layout']}
            for name,cls in [('native',GapGroupRuntime),('reuse',GapSummaryReuseRuntime),('layout',GapLayoutRuntime)]:
                rt.model[l].pool=types.MethodType(centered_pool if name=='layout' else (fused_pool if name=='reuse' else cached_pool),rt.model[l])
                rt.pool_graphs=graphs[name]
                out=cls.current_summary(rt,l)
                exact=[torch.equal(x,y) for x,y in zip(target,out)]
                errors=[float((x.float()-y.float()).abs().max()) for x,y in zip(target,out)]
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(300):cls.current_summary(rt,l)
                end.record();end.synchronize();ms=start.elapsed_time(end)/300
                row=dict(layer=l,variant=name,exact=exact,max_errors=errors,ms=ms)
                rows.append(row);print(json.dumps(row),flush=True)
        atomic('artifacts/gap_pool_20261031/center_current_checks.json',dict(rows=rows))
    finally:rt.close()
if __name__=='__main__':main()
