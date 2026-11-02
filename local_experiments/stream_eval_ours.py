"""Current accepted sparse runtime using exactly the official comparison inputs."""
import argparse,json,torch
from pathlib import Path
from .broad_common import Engine
from .gap_gain_vector64_runtime import GapGainVector64Runtime
from .learned_cache.io import digest
from .stream_eval_common import benchmark,REPO
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs={k:v.cuda() for k,v in torch.load(a.inputs,weights_only=True,map_location='cpu').items()}
    engine=Engine();rt=GapGainVector64Runtime(engine,str(REPO/'artifacts/full_route_grad_20261031/epoch1.pt'),probes=16)
    model=REPO/'artifacts/four_gaps_20261031/shared/epoch1.pt';rt.model.load_state_dict(torch.load(model,weights_only=False,map_location='cpu')['model'])
    def generate():
        rt.reset();zs=[]
        for chunk in range(41):
            x=inputs['initial_noise'][:,chunk*3:chunk*3+3]
            for step in range(4):
                clean=engine.forward(x,inputs['prompt_embeds'],chunk,step)
                if step<3:x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),inputs['transition_noise'][chunk,step].flatten(0,1),torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
            engine.forward(clean,inputs['prompt_embeds'],chunk,4);zs.append(clean)
        return torch.cat(zs,1)
    metadata=dict(method='ours',checkpoint_sha256=digest(REPO/'checkpoints/self_forcing_dmd.pt'),sparse_checkpoint_sha256=digest(model),inputs_sha256=digest(a.inputs),height=512,width=768,chunks=41,frames=480,fps=16,history_chunk_cap=5,history_pages=338,full_history_pages=360,aux_allowance_bytes=16777216,scope='Accepted GainVector64; backbone frozen, trained sparse modules; no rebase; unchanged 90% exact-page sparsity, 16 probes, 4 page-local representatives.')
    try:benchmark('ours',a.output,generate,metadata,storage=rt.finish)
    finally:rt.close()
if __name__=='__main__':main()
