"""Paired latent and decoded differences for moving QK scaling into the producer."""
import argparse
import json
import os
import numpy as np
import torch
from .accepted_tiers import config
from .deployment import configure, load_weights, make_runtime
from .weights import ROOT
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from ..stream_eval_common import native_vae, decode, save_video
from ..high_sparse_experiment import prefix_quality

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tier', required=True, choices=['base','intersection','step'])
    args = p.parse_args()
    root = ROOT / 'artifacts/video_runtime_v1/scaled_route_qk_v1'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = config(args.tier, native_route_qk=True, scale_in_qk=True, local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    model, reader, phi, basis, _ = load_weights(cfg)
    engine = Engine()
    engine.wrapper.requires_grad_(False)
    rt = make_runtime(engine, model, reader, phi, basis, cfg)
    emb = torch.load(ROOT / 'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt', weights_only=True).cuda()
    rt.reset()
    new = torch.cat([generate_chunk(engine, emb, 701, c) for c in range(9)], dim=1).cpu()
    rt.close()
    del rt, engine
    old = torch.load(root.parent / 'post_rope_round1/video' / args.tier / 'latents.pt', weights_only=True)
    torch.save(new, root / f'{args.tier}_latents.pt')
    def nmse(a,b):
        return ((a.float()-b.float()).square().sum()/b.float().square().sum()).item()
    os.environ['TORCH_HOME'] = str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    import lpips
    vae = native_vae()
    x, y = (np.concatenate(decode(vae, z.cuda(), 105)) for z in (new, old))
    metric = lpips.LPIPS(net='alex').eval().requires_grad_(False)
    quality = prefix_quality([x], [y], metric)
    result = dict(tier=args.tier, latent_nmse=nmse(new,old),
                  chunk_nmse=[nmse(a,b) for a,b in zip(new.split(3,1),old.split(3,1))],
                  max_abs=(new.float()-old.float()).abs().max().item(),
                  rgb_mae_255=float(np.abs(x.astype(np.int16)-y.astype(np.int16)).mean()), quality=quality)
    (root / f'{args.tier}_quality.json').write_text(json.dumps(result, indent=2))
    save_video(root/f'{args.tier}_comparison.mp4', [np.concatenate([y,x], axis=2)])

if __name__ == '__main__':
    main()
