"""Replay one saved 15s trajectory; separate selected-attention math and Dense error."""
import json,time
from pathlib import Path
import torch
from .weights import ROOT,load
from .runtime import VideoRuntime
from ..review_v1.ops import Config
from ..review_v1.runtime import SelectedAttention
from ..broad_common import Engine,noise
from ..learned_cache.io import atomic

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model,reader,phi,B,identity=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    engine=Engine();engine.wrapper.requires_grad_(False);rt=VideoRuntime(engine,model,reader,phi,B,full_pset=True,tile_batch=6,compile_route=True,compile_tail=True,action_kernel=True,validate=True,config=Config(history_pages=360))
    z=torch.load(ROOT/'artifacts/video_runtime_v1/formal_15_30/15s/original/latents.pt',weights_only=True).cuda();emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda();rows=[];original=SelectedAttention.fixed
    def checked(q,k,v,s,basis,sel):
        ref=original(q,k,v,s,basis,sel)
        dense=torch.nn.functional.scaled_dot_product_attention(q[None],k.flatten(1,2)[None],v.flatten(1,2)[None])[0]
        delta=ref.float()-dense.float();rows.append(dict(chunk=engine.current['chunk'],step=engine.current['step'],layer=engine.current['layer'],reference_sparse_vs_dense_nmse=float(delta.square().sum()/dense.float().square().sum()),max_abs=float(delta.abs().max())))
        return ref
    SelectedAttention.fixed=staticmethod(checked)
    try:
        for c in range(21):
            if c in (2,20):engine.forward(noise(701,c,0),emb,c,0)
            engine.forward(z[:,c*3:c*3+3],emb,c,4)
        report=dict(passed=True,identity=identity,scope='saved original15s clean history; step0 first6tiles each sparse layer; current optimized math compared against SelectedAttention reference, separately reference versus Dense on identical QKV',kernel_checks=rt.validation,approximation_checks=rows)
        atomic(ROOT/'artifacts/video_runtime_v1/formal_short_15/late_diagnostic.json',report);print(json.dumps(report))
    finally:SelectedAttention.fixed=staticmethod(original);rt.close()
if __name__=='__main__':run()
