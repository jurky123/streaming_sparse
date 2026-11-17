"""Replay the 2026-11-13 inputs with the accepted round-eight baseline."""
import argparse
import os
from pathlib import Path
import torch
from ..broad_common import Engine
from ..learned_cache.io import atomic, digest
from ..stream_eval_common import benchmark, identify_gpu_process, REPO
from ..review_v1.ops import Config
from .weights import load
from .runtime import VideoRuntime
from . import reader_reduce, reader_late, scores_moments

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--summary',choices=['old','new'],default='new');p.add_argument('--without-full-pset',action='store_true');a=p.parse_args()
    identify_gpu_process()
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    source=out/'source';source.mkdir(exist_ok=True)
    hashes={}
    for file in Path(__file__).parent.glob('*.py'):
        (source/file.name).write_bytes(file.read_bytes());hashes[file.name]=digest(file)
    inputs={k:v.cuda() for k,v in torch.load(a.inputs,weights_only=True,map_location='cpu').items()}
    summary_path=REPO/('artifacts/full_coverage_v1/training/C_s17/deployment.pt' if a.summary=='new' else 'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt')
    method='current_round8_full_pset' if a.summary=='new' and not a.without_full_pset else f'{a.summary}_summary_'+('risk' if a.without_full_pset else 'full_pset')
    model,reader,phi,B,identity=load(checkpoint=summary_path,reader_checkpoint=REPO/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    reader_reduce.WARPS=4;reader_reduce.SHARD=128;reader_reduce.AGG32=True
    reader_reduce.PACKED=True;reader_reduce.MOMENTS=True;reader_reduce.GROUP=1
    reader_reduce.FMA=False;reader_reduce.PAGE32=False;reader_reduce.reader_kernel=reader_late.reader_kernel
    scores_moments.BLOCKED=True;scores_moments.LAZY=False
    engine=Engine();engine.wrapper.requires_grad_(False)
    flags=dict(full_pset=not a.without_full_pset,tile_batch=72,compile_route=True,compile_tail=True,action_kernel=True,tensor_math=True,projection_float=False,compact_reader=True,stream_attention=True,stream_tail_mode=4,stream_warps=8,bf16_qk=True,split_summary=True,fused_risk=False,static_route=False)
    rt=VideoRuntime(engine,model,reader,phi,B,config=Config(history_pages=360),**flags)
    def generate():
        rt.reset();zs=[]
        for chunk in range(41):
            x=inputs['initial_noise'][:,chunk*3:chunk*3+3]
            for step in range(4):
                clean=engine.forward(x,inputs['prompt_embeds'],chunk,step)
                if step<3:
                    x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),inputs['transition_noise'][chunk,step].flatten(0,1),torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
            engine.forward(clean,inputs['prompt_embeds'],chunk,4);zs.append(clean)
        return torch.cat(zs,1)
    metadata=dict(method=method,summary_version=a.summary,weights=identity,flags=flags,source=hashes,checkpoint_sha256=digest(REPO/'checkpoints/self_forcing_dmd.pt'),inputs_sha256=digest(a.inputs),height=512,width=768,chunks=41,frames=480,fps=16,history_chunk_cap=5,history_pages=360,baseline='accepted round 8; round 9/10 experiments disabled',scope='Original benchmark: one full warmup, two repetitions of generation + VAE; original quality windows; no idle VRAM reservoir')
    atomic(out/'metadata.json',metadata)
    try:benchmark(method,out,generate,metadata,storage=rt.finish)
    finally:rt.close()

if __name__=='__main__':main()
