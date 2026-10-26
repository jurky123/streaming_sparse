"""Eight predeclared Val prompts, seed0: Dense/original/Fast1/Opt2 comparisons."""
import argparse,gc,json,os
from pathlib import Path
import torch
from ..common import config,atomic_json,update,paired_seed,sha
from .network import load_optimized

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory)
    out=root/'optimization2/quality';out.mkdir(exist_ok=True)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));paired_seed(0)
    manifest=json.loads((root/'e2e_manifest.json').read_text())
    for path,digest in {**manifest['code_sha256'],**manifest['frozen_files']}.items():assert sha(path)==digest,path
    def status(stage,**kw):
        update(out/f'rank{rank}.json',status=stage,rank=rank,**kw);print(stage,kw,flush=True)
    status('loading');net,_=load_optimized(root)
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
    def decode(latent):
        pixels=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return pixels
    with sdpa_backend():
        for pid in manifest['prompt_ids'][rank::world]:
            folder=out/f'{pid}.seed0';folder.mkdir(exist_ok=True)
            sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0))
            embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
            original=torch.load(root/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)
            dense=decode(original['Dense']);result=dict(prompt_id=pid,seed=0,variants={},test_used=False)
            for mode in ['K32-Set','V32-Set']:
                name='K32'if mode=='K32-Set'else'V32';status('evaluating',prompt=pid,variant=name)
                fast1=torch.load(root/f'optimization/quality_seed0/{pid}.seed0/Fast-{name}.latent.pt',map_location='cpu',weights_only=True)
                net.set_mode(mode);chunks,_=trajectory(net,embeds,pid,0,True);latent=torch.cat(chunks,1)
                assert torch.equal(latent[:,:3],original['Dense'][:,:3])
                pix=decode(latent);assert torch.equal(pix[:9],dense[:9]);prior=decode(fast1);old=decode(original[mode])
                routed=net.routing_records;assert len(routed)==348 and {r['layer']for r in routed}==set(range(1,30))
                keep=torch.stack([r['keep']for r in routed]).mean().item()
                retained=torch.stack([r['retained_H']+r['retained_C']for r in routed]).sum().item()
                denom=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in routed)
                row=dict(macro_keep=keep,weighted_keep=retained/denom,quality_vs_dense=pixel_metrics(dense,pix,metric),
                    quality_vs_fast1=pixel_metrics(prior,pix,metric),quality_vs_original=pixel_metrics(old,pix,metric),
                    latent_nmse_vs_fast1=[float((latent[:,j:j+3].float()-fast1[:,j:j+3].float()).square().sum()/fast1[:,j:j+3].float().square().sum())for j in range(0,12,3)],
                    first_chunk_latent_and_rgb_identical=True,all29_sparse_layers=True)
                torch.save(latent,folder/f'{name}.latent.pt');atomic_json(folder/f'{name}.json',row);result['variants'][name]=row
                preview(folder/f'Dense_Original_Fast1_Opt2_{name}.mp4',{'Dense':dense,'Original':old,'Fast1':prior,'Opt2':pix},['Dense','Original','Fast1','Opt2'])
                del chunks,latent,pix,prior,old;net._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
            atomic_json(folder/'result.json',result)
    assert tuple(p._version for p in net.parameters())==versions
    status('completed')

if __name__=='__main__':main()
