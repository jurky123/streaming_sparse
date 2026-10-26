"""Prove the final IO fusion preserves all 32 already-evaluated videos."""
import argparse
import json
import os
import gc
import torch
from .common import config,atomic_json,update,paired_seed
from .opt4_fused import load_fused
from .opt4.route import CompactRoute
from .opt4.network import CompactCache
from .opt4.suite import VARIANTS


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,_=config(a.directory)
    out=root/'optimization4/fused_validation';out.mkdir(exist_ok=True)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));paired_seed(0)
    update(out/f'rank{rank}.json',status='loading');net,_=load_fused(root)
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from utils.wan_wrapper import WanVAEWrapper
    vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds=ReplayDataset('val',require_complete=True);manifest=json.loads((root/'e2e_manifest.json').read_text());rows=[]
    def decode(latent):
        pixel=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return pixel
    with sdpa_backend():
        for pid in manifest['prompt_ids'][rank::world]:
            sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0))
            embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
            for variant,options in VARIANTS.items():
                route=CompactRoute(**options);net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
                gc.collect();torch.cuda.empty_cache()
                for kind in ['K32','V32']:
                    update(out/f'rank{rank}.json',status='checking',variant=variant,mode=kind,prompt=pid)
                    net.set_mode(kind+'-Set');chunks,_=trajectory(net,embeds,pid,0,True);latent=torch.cat(chunks,1)
                    prior=torch.load(root/f'optimization4/quality/{pid}.seed0/{variant}-{kind}.latent.pt',map_location='cpu',weights_only=True)
                    pixel=decode(latent);refpix=decode(prior)
                    row=dict(prompt_id=pid,variant=variant,mode=kind,latent_bit_identical=torch.equal(latent,prior),
                        rgb_bit_identical=torch.equal(pixel,refpix),latent_nmse=float((latent.float()-prior.float()).square().sum()/prior.float().square().sum()),
                        rgb_max_abs=float((pixel-refpix).abs().max()))
                    rows.append(row);print(row,flush=True);atomic_json(out/f'results_rank{rank}.json',rows)
                    del latent,chunks,pixel,refpix;net._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
    update(out/f'rank{rank}.json',status='completed',passed=all(r['latent_bit_identical']and r['rgb_bit_identical']for r in rows))


if __name__=='__main__':main()
