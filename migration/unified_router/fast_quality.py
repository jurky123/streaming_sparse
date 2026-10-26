"""Predeclared Val-only regression for execution and shared-query approximations."""
import argparse,gc,json,os,time
from pathlib import Path
import torch
from .common import config,atomic_json,update,paired_seed,sha
from .fast_network import load_fast
from .fast_route import FastRoute
from .fast_graph import GraphCache

VARIANTS=[('Fast-K32','K32-Set',1,16),('Fast-V32','V32-Set',1,16),
          ('Q16-K32','K32-Set',16,16),('Q16-V32','V32-Set',16,16),('Q64-K32','K32-Set',64,64)]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed-count',type=int,default=1);a=p.parse_args()
    root,m=config(a.directory);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));paired_seed(0)
    manifest=json.loads((root/'e2e_manifest.json').read_text())
    for path,digest in {**manifest['code_sha256'],**manifest['frozen_files']}.items():assert sha(path)==digest,path
    def status(stage,**kw):
        update(out/f'rank{rank}.json',status=stage,rank=rank,**kw);print(stage,kw,flush=True)
    status('loading');net,_=load_fast(root,graph=True,fused=True,history_cache=True,compiled_build=True)
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from migration.frozen_quality.worker import pixel_metrics,preview
    from utils.wan_wrapper import WanVAEWrapper
    import lpips
    vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).cuda().eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds=ReplayDataset('val',require_complete=True);assert ds.fingerprint()==manifest['val_fingerprint']
    versions=tuple(p._version for p in net.parameters())
    cases=[(pid,seed)for pid in manifest['prompt_ids']for seed in range(a.seed_count)][rank::world]
    def decode(latent):
        pixels=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu()
        vae.model.clear_cache();return pixels
    with sdpa_backend():
        for pid,seed in cases:
            folder=out/f'{pid}.seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():continue
            sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==seed))
            embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
            old=torch.load(root/f'videos/{pid}.seed{seed}/latents.pt',map_location='cpu',weights_only=True)
            pixels={name:decode(old[mode])for name,mode in [('Dense','Dense'),('Old-K32','K32-Set'),('Old-V32','V32-Set')]}
            result=dict(prompt_id=pid,generation_seed=seed,variants={},test_used=False)
            for name,mode,stride,group in VARIANTS:
                status('evaluating',prompt=pid,seed=seed,variant=name)
                net.query_stride=stride;net.attention_group=group
                net.fast_route=FastRoute(True,True,stride,512);net.graph_route=GraphCache(net.fast_route,512)
                net.set_mode(mode);gc.collect();torch.cuda.empty_cache()
                chunks,_=trajectory(net,embeds,pid,seed,True);latent=torch.cat(chunks,1)
                assert torch.equal(latent[:,:3],old['Dense'][:,:3])
                pix=decode(latent);assert torch.equal(pix[:9],pixels['Dense'][:9]);pixels[name]=pix
                keep=torch.stack([r['keep']for r in net.routing_records]).mean().item()
                retained=torch.stack([r['retained_H']+r['retained_C']for r in net.routing_records]).sum().item()
                denom=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in net.routing_records)
                prior='Old-K32' if mode=='K32-Set'else 'Old-V32'
                row=dict(macro_keep=keep,weighted_keep=retained/denom,
                    quality_vs_dense=pixel_metrics(pixels['Dense'],pix,metric),
                    quality_vs_old=pixel_metrics(pixels[prior],pix,metric),
                    latent_nmse_vs_old=[float((latent[:,j:j+3].float()-old[mode][:,j:j+3].float()).square().sum()/old[mode][:,j:j+3].float().square().sum())for j in range(0,12,3)])
                result['variants'][name]=row;atomic_json(folder/f'{name}.json',row)
                torch.save(latent,folder/f'{name}.latent.pt')
                net._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
            preview(folder/'Dense_OldK_FastK_Q16K_Q64K.mp4',pixels,['Dense','Old-K32','Fast-K32','Q16-K32','Q64-K32'])
            preview(folder/'Dense_OldV_FastV_Q16V.mp4',pixels,['Dense','Old-V32','Fast-V32','Q16-V32'])
            atomic_json(folder/'result.json',result)
            del pixels,old,chunks,latent;gc.collect()
    assert tuple(p._version for p in net.parameters())==versions
    status('completed',cases=len(cases))

if __name__=='__main__':main()
