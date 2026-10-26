"""Sealed best checkpoints, closed-loop latents, decoded RGB and paired videos."""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import time
import torch

from .common import ROOT,config,atomic_json,sha,save,paired_seed,update


def seal(root,m):
    from streaming_sparse.v08.replay import ReplayDataset
    ds=ReplayDataset('val',require_complete=True)
    assert ds.fingerprint()==m['replay_fingerprints']['val']
    assert m['layers']==list(range(1,30))and m['budget_locked']
    assert json.loads((root/'network_audit.json').read_text())['passed']
    files={};selection=[];seed=m['end_to_end']['training_seed']
    for layer in m['layers']:
        for path in [root/f'action_basis/layer{layer:02d}.pt',root/f'normalization/layer{layer:02d}.pt']:
            files[str(path)]=sha(path)
        for kind in m['groups']:
            directory=root/f'P3_training/layer{layer:02d}/{kind}/seed{seed}'
            val=json.loads((directory/'best_validation.json').read_text())
            curve=json.loads((directory/'validation.json').read_text())['curve']
            assert val['nmse']==min(r['nmse']for r in curve)
            files[str(directory/'best.pt')]=sha(directory/'best.pt')
            selection.append(dict(layer=layer,kind=kind,seed=seed,updates=val['updates'],nmse=val['nmse']))
    names=['network.py','core.py','learning.py','video_worker.py','video_report.py']
    code={str(ROOT/'migration/unified_router'/name):sha(ROOT/'migration/unified_router'/name)for name in names}
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),beta=m['monitor_beta'],
        training_seed=seed,prompt_ids=sorted(ds.registry['splits']['val']),generation_seeds=[0,1],
        modes=['Dense','U-Sol','K32-Set','V32-Set'],sparse_layers=m['layers'],dense_layers=[0],
        frozen_files=files,code_sha256=code,selections=selection,teacher_sha256=m['teacher_sha256'],
        val_fingerprint=ds.fingerprint(),test_used=False,selection_locked_before_video=True,
        budget_scope='Subsequent sparse chunks; first chunk and context writes Dense.',
        router_float32_tf32=False,
        resolution=[480,832],latent_frames=12,rgb_frames=45,fps=16)
    path=root/'e2e_manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert all(old[k]==manifest[k]for k in ['frozen_files','code_sha256','beta','prompt_ids','modes'])
        return
    atomic_json(path,manifest)
    print(json.dumps(dict(sealed_candidates=2,layers=29,cases=len(manifest['prompt_ids'])*2,beta=manifest['beta'])),flush=True)


@torch.no_grad()
def worker(root,m):
    from .network import load_network
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.v08.common import nmse
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from migration.frozen_quality.worker import pixel_metrics,preview
    import lpips
    from utils.wan_wrapper import WanVAEWrapper
    torch.cuda.set_device(0);paired_seed(0)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    manifest=json.loads((root/'e2e_manifest.json').read_text())
    for name,digest in {**manifest['frozen_files'],**manifest['code_sha256']}.items():assert sha(name)==digest,name
    ds=ReplayDataset('val',require_complete=True);assert ds.fingerprint()==manifest['val_fingerprint']
    cases=[(p,s)for p in manifest['prompt_ids']for s in manifest['generation_seeds']][rank::world]
    status=root/f'N2_videos/rank{rank}.json'
    update(status,status='loading',completed=0,total=len(cases),pid=os.getpid())
    net,_=load_network(root,manifest['training_seed'])
    versions=tuple(p._version for p in net.model.parameters())
    metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).cuda().eval().requires_grad_(False)
    vae=WanVAEWrapper().cuda().to(dtype=torch.bfloat16).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert abs(float(metric(torch.zeros(1,3,64,64,device='cuda'),torch.zeros(1,3,64,64,device='cuda'))))<1e-7
    rows=[]
    with sdpa_backend():
        for index,(pid,seed)in enumerate(cases,1):
            if(root/'STOP').exists():raise RuntimeError('Stop requested')
            folder=root/'videos'/f'{pid}.seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():
                rows.append(json.loads((folder/'result.json').read_text()));continue
            sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==seed))
            embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
            pixels={};latents={};details={}
            for mode in manifest['modes']:
                update(status,status='evaluating',completed=index-1,total=len(cases),mode=mode,prompt_id=pid,generation_seed=seed,pid=os.getpid())
                net.set_mode(mode);started=time.monotonic()
                chunks,diagnostics=trajectory(net,embeds,pid,seed,mode!='Dense')
                latents[mode]=torch.cat(chunks,1)
                route=list(net.routing_records)
                if mode!='Dense':
                    assert torch.equal(chunks[0],latents['Dense'][:,:3]),'First Dense chunk changed'
                    expected={(start,step,layer)for start in [3,6,9]for step in range(4)for layer in range(1,30)}
                    assert len(route)==348 and {(r['start'],r['step'],r['layer'])for r in route}==expected
                    assert all(0<r['keep']<=1 for r in route)
                else:assert not route
                pixels[mode]=vae.decode_to_pixel(latents[mode].cuda().to(torch.bfloat16),use_cache=False)[0].cpu()
                vae.model.clear_cache();assert pixels[mode].shape==(45,3,480,832)
                details[mode]=dict(diagnostics=diagnostics,sparsity=route,elapsed_seconds=time.monotonic()-started)
                if mode!='Dense':
                    assert torch.equal(pixels[mode][:9],pixels['Dense'][:9]),'First Dense RGB frames changed'
                    details[mode]['quality']=pixel_metrics(pixels['Dense'],pixels[mode],metric)
                    total=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in route)
                    details[mode]['budget']=dict(macro_keep=sum(r['keep']for r in route)/len(route),
                        token_weighted_keep=sum(r['retained_H']+r['retained_C']for r in route)/total,
                        history_keep=sum(r['retained_H']for r in route)/sum(r['queries']*r['history_tokens']for r in route),
                        current_keep=sum(r['retained_C']for r in route)/sum(r['queries']*r['current_tokens']for r in route))
                    details[mode]['latent_chunks']=[dict(chunk=i,nmse=float(nmse(c,latents['Dense'][:,i*3:(i+1)*3])))for i,c in enumerate(chunks)]
                atomic_json(folder/f'{mode}_metrics.json',details[mode])
                net._eval_history=None;net._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
            save(folder/'latents.pt',latents)
            preview(folder/'Dense_K32_V32.mp4',pixels,['Dense','K32-Set','V32-Set'])
            preview(folder/'Dense_Usol_V32.mp4',pixels,['Dense','U-Sol','V32-Set'])
            row=dict(prompt_id=pid,generation_seed=seed,metrics=details,
                comparison=str(folder/'Dense_K32_V32.mp4'),baseline_comparison=str(folder/'Dense_Usol_V32.mp4'),
                first_dense_frames_identical=True)
            atomic_json(folder/'result.json',row);rows.append(row)
            atomic_json(root/f'N2_videos/rows_rank{rank}.json',dict(rows=rows))
            update(status,status='evaluating',completed=index,total=len(cases),pid=os.getpid())
            del pixels,latents,chunks;gc.collect();torch.cuda.empty_cache()
    assert tuple(p._version for p in net.model.parameters())==versions
    assert not any(p.requires_grad for p in net.parameters())
    update(status,status='completed',completed=len(cases),total=len(cases),backbone_unchanged=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--seal',action='store_true')
    a=p.parse_args();root,m=config(a.directory)
    (seal if a.seal else worker)(root,m)


if __name__=='__main__':main()
