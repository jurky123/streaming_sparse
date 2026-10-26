"""Matched eight-prompt bitwise latent/RGB check for the conservative adapter."""
import argparse,json,os
from pathlib import Path
import torch
from .common import config,atomic_json,update,paired_seed,sha
from .opt2_strict import load_strict

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory)
    out=root/'optimization2/strict_validation';out.mkdir(exist_ok=True);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    paired_seed(0);update(out/f'rank{rank}.json',status='loading')
    net,_=load_strict(root)
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from utils.wan_wrapper import WanVAEWrapper
    vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds=ReplayDataset('val',require_complete=True);manifest=json.loads((root/'e2e_manifest.json').read_text());rows=[]
    with sdpa_backend():
        for pid in manifest['prompt_ids'][rank::world]:
            sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0))
            embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
            for name,mode in [('K32','K32-Set'),('V32','V32-Set')]:
                update(out/f'rank{rank}.json',status='checking',prompt=pid,variant=name)
                net.set_mode(mode);chunks,_=trajectory(net,embeds,pid,0,True);got=torch.cat(chunks,1)
                prior=torch.load(root/f'optimization/quality_seed0/{pid}.seed0/Fast-{name}.latent.pt',map_location='cpu',weights_only=True)
                pixel=vae.decode_to_pixel(got.cuda(),use_cache=False)[0].cpu();vae.model.clear_cache()
                ref_pixel=vae.decode_to_pixel(prior.cuda(),use_cache=False)[0].cpu();vae.model.clear_cache()
                row=dict(prompt_id=pid,mode=name,latent_bit_identical=torch.equal(got,prior),rgb_bit_identical=torch.equal(pixel,ref_pixel),
                    latent_nmse=float((got.float()-prior.float()).square().sum()/prior.float().square().sum()),
                    rgb_max_abs=float((pixel-ref_pixel).abs().max()))
                rows.append(row);print(json.dumps(row),flush=True);atomic_json(out/f'results_rank{rank}.json',rows)
    update(out/f'rank{rank}.json',status='completed',passed=all(r['latent_bit_identical']and r['rgb_bit_identical']for r in rows))

if __name__=='__main__':main()
