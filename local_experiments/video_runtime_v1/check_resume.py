"""Rebuild KV/encodings from clean latents and verify exact continuation."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from .weights import load,ROOT
from .runtime import VideoRuntime
from .worker import generate
from .queue import folder
from ..broad_common import Engine
from ..learned_cache.io import atomic

@torch.no_grad()
def run(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=torch.load(args.source,map_location='cpu',weights_only=True);chunks=source.shape[1]//3
    if not 0<args.prefix<chunks:raise ValueError('prefix must leave at least one chunk to regenerate')
    job=dict(case='city_walk',noise_seed=701,method='c_pset',chunks=chunks)
    root=Path(args.output);out=folder(root,job)
    if (out/'generation.json').exists():raise ValueError('Use a fresh check output directory')
    for i in range(args.prefix):atomic(out/f'chunk{i:02d}.pt',source[:,i*3:i*3+3],True)
    model,reader,phi,B,identity=load();engine=Engine();engine.wrapper.requires_grad_(False)
    rt=VideoRuntime(engine,model,reader,phi,B,full_pset=True,tile_batch=6)
    queue=SimpleNamespace(root=root,update=lambda *unused:None)
    try:
        generate(queue,'resume-check',job,rt,identity,'resume-check')
        actual=torch.load(out/'latents.pt',weights_only=True);assert torch.equal(actual,source),'Clean replay changed continuation'
        atomic(root/'resume_check.json',dict(passed=True,bitwise_equal=True,prefix=args.prefix,total_chunks=chunks,cache_rebuilt='raw KV + immutable summaries + Reader/Phi encodings'))
        print('Clean replay continuation bitwise equal',flush=True)
    finally:rt.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='artifacts/video_runtime_v1/smoke/latents.pt');p.add_argument('--output',default='artifacts/video_runtime_v1/resume_check');p.add_argument('--prefix',type=int,default=2);run(p.parse_args())
