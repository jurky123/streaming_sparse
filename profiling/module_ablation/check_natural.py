"""Pair original and natural paging with the frozen five-prompt dense references."""
import sys,json
from pathlib import Path
import numpy as np
import torch
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/'conditional_scope'))
from neighbors import CASES,SEEDS,embpath,root,scope,clean,repo,config,configure,Engine,make_runtime,load_weights,attach,decode,native_vae,prefix_quality,write
from interventions import install
import lpips
@torch.no_grad()
def run(case):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('intersection_fast',local_assets=repo);configure(cfg)
    engine=Engine()
    for b in engine.wrapper.model.blocks:b.self_attn.sink_size=1
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    rt.sparse_clean=True
    gen=attach(rt,engine,10);scope(rt,dict(name='old'));install(rt,case)
    vae=native_vae();metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    for p in CASES:
        emb=torch.load(embpath(p),weights_only=True).cuda()
        for seed in SEEDS:
            dest=here/('compensation_validation' if '--compensation' in sys.argv else 'natural_validation')/case/f'{p}_{seed}.json'
            if dest.exists():continue
            ref=np.load(root/'validation'/f'{p}_s{seed}'/'dense.npy',mmap_mode='r')
            rt.reset()
            qs=prefix_quality(decode(vae,gen(emb,seed,9),105),[ref],metric)
            write(dest,qs)
run(sys.argv[1])
