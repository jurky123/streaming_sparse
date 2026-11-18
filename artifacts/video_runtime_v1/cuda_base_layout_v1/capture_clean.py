"""Capture one actual P432 clean invocation from the deployed base."""
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.weights import ROOT

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    original=engine.dense
    path=Path('/tmp/jzk_cuda_clean_input.pt')
    def capture(q,k,v):
        cur=engine.current
        if cur['step']==4 and cur['layer']==1 and k.shape[1]==27648:
            torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),chunk=cur['chunk']),path)
            engine.dense=original
        return original(q,k,v)
    engine.dense=capture
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    rt.reset()
    for c in range(6):generate_chunk(engine,emb,701,c)
    rt.close()
    assert path.exists()
if __name__=='__main__':main()
