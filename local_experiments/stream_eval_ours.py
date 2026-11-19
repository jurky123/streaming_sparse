"""Current accepted sparse runtime using exactly the official comparison inputs."""
import argparse,json,torch
from pathlib import Path
from .broad_common import Engine
from .review_v1.model import Model
from .review_v1.ablation import AblationRuntime
from .review_v1.ops import Config
from .learned_cache.io import digest
from .stream_eval_common import benchmark,REPO
@torch.no_grad()
def main():
    from local_experiments.stream_eval_common import identify_gpu_process
    identify_gpu_process()
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs={k:v.cuda() for k,v in torch.load(a.inputs,weights_only=True,map_location='cpu').items()}
    engine=Engine()
    base=REPO/'artifacts/four_gaps_20261031/shared/epoch1.pt'
    model=REPO/'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'
    network=Model(str(base))
    network.load_state_dict(torch.load(model,map_location='cuda',weights_only=False)['model'])
    old=torch.load(base,map_location='cpu',weights_only=False)['model']
    centers={l:old[f'layers.{l}.center'].cuda() for l in range(1,30)}
    rt=AblationRuntime(engine,network,'no_downstream',centers,config=Config(history_pages=360))
    def generate():
        rt.reset();zs=[]
        for chunk in range(41):
            x=inputs['initial_noise'][:,chunk*3:chunk*3+3]
            for step in range(4):
                clean=engine.forward(x,inputs['prompt_embeds'],chunk,step)
                if step<3:x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),inputs['transition_noise'][chunk,step].flatten(0,1),torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
            engine.forward(clean,inputs['prompt_embeds'],chunk,4);zs.append(clean)
        return torch.cat(zs,1)
    metadata=dict(method='ours',checkpoint_sha256=digest(REPO/'checkpoints/self_forcing_dmd.pt'),sparse_checkpoint_sha256=digest(model),inputs_sha256=digest(a.inputs),height=512,width=768,chunks=41,frames=480,fps=16,history_chunk_cap=5,history_pages=360,full_history_pages=360,aux_allowance_bytes=16777216,scope='Current summary_pretrained; 360 history pages (15 latent frames) for equal history window; summary overhead reported separately; FP32 reference implementation.')
    def storage():
        value=rt.finish();extra=sum(x.numel()*x.element_size() for x in centers.values())
        value['ablation_center_bytes']=extra;value['persistent_bytes']+=extra
        value['within_limit']=value['persistent_bytes']<=value['limit_bytes']
        value['protocol']='equal 360-page history; summary overhead is additional, no equal-memory claim'
        return value
    try:benchmark('ours',a.output,generate,metadata,storage=storage)
    finally:rt.close()
if __name__=='__main__':main()
