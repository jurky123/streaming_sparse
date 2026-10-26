"""Fresh paired all29-layer closed-loop videos, metrics, and generation timings."""
import argparse,gc,json,os,time,statistics
from pathlib import Path
import torch
from .setup import ROOT
from .network import load
from ..common import config,atomic_json,update,paired_seed,sha
from ..speed_profile import gpu_info

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','final'],required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));paired_seed(0)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert json.loads((ROOT/'graph_preflight.json').read_text())['passed']
    out=ROOT/('e2e_before' if a.phase=='before' else 'e2e_final');out.mkdir(exist_ok=True)
    def status(stage,**kw):update(out/f'rank{rank}.json',status=stage,**kw);print(stage,kw,flush=True)
    status('loading');net,m=load()
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from migration.frozen_quality.worker import pixel_metrics,preview
    from utils.wan_wrapper import WanVAEWrapper
    import lpips
    vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    metric=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    ds=ReplayDataset('val',require_complete=True)
    manifest=json.loads((Path(m['parent_experiment'])/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint']
    names=['Tile15_before','Tile20_before'] if a.phase=='before' else ['Tile15_before','Tile15_after','Tile20_before','Tile20_after']
    pids=manifest['prompt_ids'][:a.limit] if a.limit else manifest['prompt_ids']
    def generate(name,embeds,pid):
        net.select(name);chunks,_=trajectory(net,embeds,pid,0,name!='Dense');return torch.cat(chunks,1)
    def decode(z):
        pixels=vae.decode_to_pixel(z.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return pixels
    versions=tuple(x._version for x in net.model.parameters())
    with sdpa_backend():
        for pid in pids[rank::world]:
            folder=out/f'{pid}.seed0';folder.mkdir(exist_ok=True)
            if (folder/'completed.json').exists():continue
            sample=ds.load(next(i for i,r in enumerate(ds.rows) if r['prompt_id']==pid and r['generation_seed']==0))
            embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
            status('quality',prompt=pid,variant='Dense');dense=generate('Dense',embeds,pid).cpu()
            torch.save(dense,folder/'Dense.latent.pt');pixels={'Dense':decode(dense)};results={};timings=[]
            for name in names:
                status('quality',prompt=pid,variant=name);z=generate(name,embeds,pid).cpu()
                assert torch.equal(z[:,:3],dense[:,:3]),(pid,name,'first chunk mismatch')
                assert len(net.routing_records)==348 and {r['layer'] for r in net.routing_records}==set(range(1,30))
                records=net.routing_records
                den=sum(r['queries']*(r['history_tokens']+r['current_tokens']) for r in records)
                kept=torch.stack([r['retained_H']+r['retained_C'] for r in records]).sum().item()
                executed=sum(r['executed_pairs'] for r in records)/den
                torch.save(z,folder/(name+'.latent.pt'));pixels[name]=decode(z)
                assert torch.equal(pixels[name][:9],pixels['Dense'][:9])
                results[name]=dict(variant=name,weighted_keep=kept/den,executed_attention_ratio=executed,
                    first_chunk_identical=True,sparse_layers=list(range(1,30)),checkpoints=dict(net.tile_checksums),
                    quality_vs_dense=pixel_metrics(pixels['Dense'],pixels[name],metric))
                atomic_json(folder/(name+'.json'),results[name])
                preview(folder/(name+'.mp4'),pixels,[name])
                # The quality run warms this variant's shape-specific graphs.
                # Alternate it with fresh Dense repeats, without reloading weights.
                for repeat in range(0 if a.phase=='before' else 4):
                    for mode in ([name,'Dense'] if repeat%2==0 else ['Dense',name]):
                        net.select(mode);gc.collect();torch.cuda.synchronize()
                        status('warmup' if repeat==0 else 'timing',prompt=pid,variant=mode,repeat=repeat)
                        start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,mode!='Dense');torch.cuda.synchronize();seconds=time.perf_counter()-start
                        actual=torch.cat(chunks,1).cpu();expected=dense if mode=='Dense' else z
                        assert torch.equal(actual,expected),(pid,mode,repeat,'non-repeatable latent')
                        timings.append(dict(variant=mode,paired_with=name,repeat=repeat,generation_seconds=seconds,repeat_identical=True))
                        del chunks,actual
                atomic_json(folder/'timings.json',dict(records=timings,gpu=gpu_info(),excludes_vae=True,includes_history_rebuild=True))
            preview(folder/'Dense.mp4',pixels,['Dense'])
            if a.phase=='before':preview(folder/'Dense_Tile15_Tile20_before.mp4',pixels,['Dense','Tile15_before','Tile20_before'])
            else:
                for variant in ['Tile15','Tile20']:
                    results[variant+'_after']['quality_vs_before']=pixel_metrics(pixels[variant+'_before'],pixels[variant+'_after'],metric)
                    preview(folder/(variant+'_Dense_before_after.mp4'),pixels,['Dense',variant+'_before',variant+'_after'])
                preview(folder/'Dense_Tile15_Tile20_after.mp4',pixels,['Dense','Tile15_after','Tile20_after'])
            times={name:statistics.median(r['generation_seconds'] for r in timings if r['variant']==name and r['repeat']>0) for name in ['Dense']+names} if timings else {}
            atomic_json(folder/'result.json',dict(prompt_id=pid,seed=0,variants=results,timings=times,scope='all29 sparse; layer0/firstchunk dense;45frames480x832;LPIPS last36frames'))
            atomic_json(folder/'completed.json',dict(status='completed',variants=names))
            del pixels,dense,z;gc.collect()
    assert versions==tuple(x._version for x in net.model.parameters())
    status('completed')

if __name__=='__main__':main()
