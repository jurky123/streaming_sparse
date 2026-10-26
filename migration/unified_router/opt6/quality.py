"""Matched Val videos and same-QKV attention audits against the accepted FP16 baseline."""
import argparse
import gc
import json
import os
import torch
from ..common import config,atomic_json,update,paired_seed,sha
from ..opt2.attention import attention
from .network import load_head
from ..opt5.network import OptimizedRoute,OPTIONS as BASE_OPTIONS


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory)
    out=root/'optimization6/quality';out.mkdir(exist_ok=True)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));paired_seed(0)
    manifest=json.loads((root/'e2e_manifest.json').read_text())
    seal=json.loads((root/'optimization6/manifest.json').read_text())
    for path,digest in {**manifest['code_sha256'],**manifest['frozen_files'],**seal['code']}.items():assert sha(path)==digest,path
    def status(stage,**kw):update(out/f'rank{rank}.json',status=stage,rank=rank,**kw);print(stage,kw,flush=True)
    status('loading');net,_=load_head(root)
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
    ref=OptimizedRoute(**BASE_OPTIONS)
    def decode(latent):
        pix=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return pix
    audit=[]
    def observe(layer,start,step,q,k,v,sketch,prepared,mask,y):
        if layer not in [1,24,29] or step!=3:return
        rr=torch.cat([ref(layer,q[:,i:i+512],sketch,prepared,step,net.router)for i in range(0,q.shape[1],512)],1)
        yr=attention(q,k,v,rr)
        yd=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]
        assert torch.isfinite(y).all()
        audit.append(dict(layer=layer,start=start,history=sketch.history_tokens,queries=q.shape[0]*q.shape[1],
            changed_queries=int((mask!=rr).any(-1).sum()),
            delta_sse=float((y.float()-yr.float()).square().sum()),reference_energy=float(yr.float().square().sum()),
            sse_to_dense=float((y.float()-yd.float()).square().sum()),
            reference_sse_to_dense=float((yr.float()-yd.float()).square().sum()),dense_energy=float(yd.float().square().sum())))
    net.inspect_callback=observe
    with sdpa_backend():
        for pid in manifest['prompt_ids'][rank::world]:
            folder=out/f'{pid}.seed0';folder.mkdir(exist_ok=True)
            sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0))
            embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
            old=torch.load(root/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)
            dense=decode(old['Dense']);panels={};references={}
            for kind in ['K32','V32']:
                latent=torch.load(root/f'optimization5/quality/{pid}.seed0/Opt5-{kind}.latent.pt',map_location='cpu',weights_only=True)
                references[kind]=(latent,decode(latent));panels[kind]={'Dense':dense,'Opt5':references[kind][1]}
            result=dict(prompt_id=pid,seed=0,variants={},test_used=False)
            for variant in ['Opt6']:
                gc.collect();torch.cuda.empty_cache()
                for kind in ['K32','V32']:
                    status('evaluating',prompt=pid,variant=variant,mode=kind)
                    audit.clear();net.set_mode(kind+'-Set');chunks,_=trajectory(net,embeds,pid,0,True);latent=torch.cat(chunks,1)
                    assert torch.equal(latent[:,:3],old['Dense'][:,:3])
                    pix=decode(latent);assert torch.equal(pix[:9],dense[:9])
                    prior,priorpix=references[kind]
                    records=net.routing_records
                    assert len(records)==348 and {r['layer']for r in records}==set(range(1,30))
                    retained=torch.stack([r['retained_H']+r['retained_C']for r in records]).sum().item()
                    denom=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in records)
                    row=dict(variant=variant,mode=kind,weighted_keep=retained/denom,
                        quality_vs_dense=pixel_metrics(dense,pix,metric),quality_vs_opt5=pixel_metrics(priorpix,pix,metric),
                        latent_nmse_by_chunk=[float((latent[:,j:j+3].float()-prior[:,j:j+3].float()).square().sum()/prior[:,j:j+3].float().square().sum())for j in range(0,12,3)],
                        latent_bit_identical=torch.equal(latent,prior),first_chunk_identical=True,attention_audit=list(audit))
                    key=variant+'-'+kind;result['variants'][key]=row
                    atomic_json(folder/(key+'.json'),row);torch.save(latent,folder/(key+'.latent.pt'))
                    panels[kind][variant]=pix
                    del chunks,latent;net._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
            for kind in ['K32','V32']:
                preview(folder/f'Dense_Opt5_Opt6_{kind}.mp4',panels[kind],['Dense','Opt5','Opt6'])
            atomic_json(folder/'result.json',result)
    assert tuple(p._version for p in net.parameters())==versions
    status('completed')


if __name__=='__main__':main()
