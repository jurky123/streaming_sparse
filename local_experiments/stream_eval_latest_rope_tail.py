"""Current accepted sparse runtime using exactly the official comparison inputs."""
import argparse,json,torch
from pathlib import Path
from .broad_common import Engine
from .review_v1.model import Model
from .review_v1.rope_runtime import RopeTailLatestRuntime
from .review_v1.ops import Config
from .learned_cache.io import digest
from .stream_eval_common import benchmark,REPO
@torch.no_grad()
def main():
    from local_experiments.stream_eval_common import identify_gpu_process
    identify_gpu_process()
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--validate',action='store_true');p.add_argument('--chunks',type=int,default=41);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs={k:v.cuda() for k,v in torch.load(a.inputs,weights_only=True,map_location='cpu').items()}
    engine=Engine()
    base=REPO/'artifacts/four_gaps_20261031/shared/epoch1.pt'
    model=REPO/'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'
    network=Model(str(base))
    network.load_state_dict(torch.load(model,map_location='cuda',weights_only=False)['model'])
    old=torch.load(base,map_location='cpu',weights_only=False)['model']
    centers={l:old[f'layers.{l}.center'].cuda() for l in range(1,30)}
    rt=RopeTailLatestRuntime(engine,network,'no_downstream',centers,config=Config(history_pages=360),validate=a.validate)
    def generate():
        rt.reset();zs=[]
        for chunk in range(a.chunks):
            x=inputs['initial_noise'][:,chunk*3:chunk*3+3]
            for step in range(4):
                clean=engine.forward(x,inputs['prompt_embeds'],chunk,step)
                if step<3:x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),inputs['transition_noise'][chunk,step].flatten(0,1),torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
            engine.forward(clean,inputs['prompt_embeds'],chunk,4);zs.append(clean)
        return torch.cat(zs,1)
    metadata=dict(method='ours_latest_rope_tail',checkpoint_sha256=digest(REPO/'checkpoints/self_forcing_dmd.pt'),sparse_checkpoint_sha256=digest(model),inputs_sha256=digest(a.inputs),height=512,width=768,chunks=41,frames=480,fps=16,history_chunk_cap=5,history_pages=360,full_history_pages=360,aux_allowance_bytes=16777216,implementation='RopeTailLatestRuntime',benchmark_script_sha256=digest(__file__),scope='Latest summary_pretrained with same checkpoint, grouping, route and summary equations; efficient page64 and latest full-mean/detail kernels; batched fused FP32 four-risk route, original graph summaries, unchanged preceding faster-tail candidate, KV scratch reuse and original grouping/selection CUDA Graphs. Current summary_pretrained; 360 history pages (15 latent frames) for equal history window; summary overhead reported separately; same BF16 operands, FP32 accumulation; floating point order differs.')
    def storage():
        value=rt.finish();extra=sum(x.numel()*x.element_size() for x in centers.values())
        value['ablation_center_bytes']=extra;value['persistent_bytes']+=extra
        value['within_limit']=value['persistent_bytes']<=value['limit_bytes']
        value['protocol']='equal 360-page history; summary overhead is additional, no equal-memory claim'
        return value
    try:
        if a.smoke:
            from .learned_cache.io import atomic
            z=generate();assert torch.isfinite(z).all()
            atomic(Path(a.output)/'latents.pt',z.cpu(),True)
            atomic(Path(a.output)/'smoke.json',dict(status='passed',shape=list(z.shape),storage=storage(),metadata=metadata))
        else:
            assert a.chunks==41 and not a.validate
            benchmark('ours_latest_rope_tail',a.output,generate,metadata,storage=storage)
    finally:rt.close()
if __name__=='__main__':main()
