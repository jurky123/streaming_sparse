"""Single real block same-input graph/eager comparison, no video decode."""
import json
from pathlib import Path
import torch
from .weights import ROOT,load
from .runtime import VideoRuntime
from .block_graph import BlockGraphs
from ..review_v1.ops import Config
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from .benchmark import measure

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model,reader,phi,B,identity=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=VideoRuntime(engine,model,reader,phi,B,full_pset=True,tile_batch=24,compile_route=True,compile_tail=True,action_kernel=True,tensor_math=True,config=Config(history_pages=360))
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    for c in range(2):generate_chunk(engine,emb,701,c)
    bg=BlockGraphs(engine,rt,layers=[15]);graph_forward=engine.wrapper.model.blocks[15].forward;rows=[]
    def checked(x,**kwargs):
        if engine.current['step']==4:return graph_forward(x,**kwargs)
        ref=bg.original[15](x,**kwargs).clone();out=graph_forward(x,**kwargs)
        error=out-ref
        rows.append(dict(step=engine.current['step'],nmse=float(error.float().square().sum()/ref.float().square().sum()),max_abs=float(error.abs().max())))
        if len(rows)==1:
            before=measure(lambda:bg.original[15](x,**kwargs),15)
            after=measure(lambda:graph_forward(x,**kwargs),15)
            rows[-1].update(before=before,after=after)
        return out
    engine.wrapper.model.blocks[15].forward=checked
    try:
        generate_chunk(engine,emb,701,2)
        result=dict(rows=rows,identity=identity,passed=all(r['max_abs']==0 for r in rows))
        Path('artifacts/video_runtime_v1/hotspot_round5/block_graph.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:bg.close();rt.close()
if __name__=='__main__':run()
