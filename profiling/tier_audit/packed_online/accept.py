"""Pair each prompt/seed under alternating baseline and SM80 variable CUDA."""
import json,sys,time
from pathlib import Path
import torch
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from queue_policy import attach
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
@torch.no_grad()
def main(method):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config(method+'_fast',local_assets=repo)
    configure(cfg)
    engine=Engine()
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    generate=attach(rt,engine,10) if method=='intersection' else lambda e,s,n:torch.cat([generate_chunk(engine,e,s,c) for c in range(n)],1)
    embeddings=[torch.load(p,weights_only=True).cuda() for p in (Path('/tmp/quota_prompt720.pt'),repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt')]
    rows=[]
    for j,(i,seed) in enumerate([(0,701),(0,701),(1,701),(0,702),(1,702)]):
        samples={}
        for fast in ((False,True) if j%2==0 else (True,False)):
            rt.set_cuda_fixed15(fast,variable=fast)
            rt.reset()
            torch.cuda.synchronize()
            t=time.perf_counter()
            z=generate(embeddings[i],seed,7)
            torch.cuda.synchronize()
            samples[fast]=(time.perf_counter()-t,z)
            assert z.isfinite().all()
        t0,ref=samples[False]
        t1,z=samples[True]
        diff=z.float()-ref.float()
        row=dict(prompt=720+i,seed=seed,baseline=t0,fast=t1,latent_nmse=(diff.square().sum()/ref.float().square().sum()).item(),max_abs=diff.abs().max().item())
        print(row,flush=True)
        if j:
            rows.append(row)
            Path(__file__).with_name(method+'_accept.json').write_text(json.dumps(rows,indent=2))
        if i==1 and seed==701:
            for idx,latent in enumerate((ref,z)):
                torch.save(latent.cpu(),f'/tmp/{method}_accept_{idx}.pt')
main(sys.argv[1])
