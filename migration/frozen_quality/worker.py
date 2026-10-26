"""Read-only paired Val replay and closed-loop RGB assessment of frozen routers."""
import argparse
from datetime import datetime,timezone
import gc
import json
import math
import os
from pathlib import Path
import time
import types
import torch
from streaming_sparse.v08.common import atomic_json,paired_seed,sha,source_identity,nmse,save,tensor_digest


def reset(network,name,routers,manifest):
    network.router=routers['mass'if name=='Dense'else name]
    network.router_version='none'if name=='Dense'else manifest['checkpoints'][name]['sha256']
    network._eval_history=None;network._eval_sketch.clear();network._parameter_versions=None


def pixel_metrics(dense,pred,metric):
    rows=[]
    for frame,(a,b)in enumerate(zip(dense,pred)):
        mse=float(((a.double()-b.double())*.5).square().mean())
        lp=float(metric(a[None].cuda(),b[None].cuda()))
        assert math.isfinite(mse)and math.isfinite(lp)
        rows.append(dict(frame=frame,mse_rgb01=mse,psnr_db=-10*math.log10(mse)if mse else None,lpips=lp,identical=mse==0))
    def summary(part):
        mse=sum(x['mse_rgb01']for x in part)/len(part)
        return dict(mse_rgb01=mse,psnr_db=-10*math.log10(mse)if mse else None,lpips=sum(x['lpips']for x in part)/len(part),frames=len(part),identical=mse==0)
    return dict(full=summary(rows),sparse_frames_only=summary(rows[9:]),chunks=[dict(chunk=i+1,**summary(rows[a:b]))for i,(a,b)in enumerate([(0,9),(9,21),(21,33),(33,45)])],per_frame=rows)


def preview(path,pixels,names):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw
    with imageio.get_writer(path,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18'])as writer:
        for frame in range(45):
            arrays=[((pixels[n][frame].permute(1,2,0)+1)*127.5).clamp(0,255).to(torch.uint8).numpy()for n in names]
            canvas=Image.fromarray(np.concatenate(arrays,axis=1));draw=ImageDraw.Draw(canvas)
            for i,n in enumerate(names):draw.text((i*832+8,8),n,fill='white',stroke_width=2,stroke_fill='black')
            writer.append_data(np.asarray(canvas))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args();root=Path(args.directory).resolve()
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));torch.cuda.set_device(0);paired_seed(0)
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['source_sha256']==source_identity()
    work=root/f'rank{rank}';work.mkdir(exist_ok=True)
    state=dict(status='loading',rank=rank,pid=os.getpid(),completed=0,started_at=datetime.now(timezone.utc).isoformat())
    def status(**values):
        state.update(values,updated_at=datetime.now(timezone.utc).isoformat());atomic_json(work/'status.json',state)
        print(json.dumps(values),flush=True)
    status(status='loading')
    from streaming_sparse.v08.execution import accelerate
    from migration.runtime import router_fast
    from streaming_sparse.v08.evaluate import load_network
    from streaming_sparse.v08.router import InformationRouter
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.v08.metrics import summarize
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from streaming_sparse.v08.launcher.worker import require_execution_audit
    require_execution_audit();accelerate.install();router_fast.install()
    model,state0=load_network(manifest['checkpoints']['mass']['path']);model.requires_grad_(False).eval()
    routers={}
    for name,info in manifest['checkpoints'].items():
        assert sha(info['path'])==info['sha256'];checkpoint=torch.load(info['path'],map_location='cpu',weights_only=True)
        router=InformationRouter(info['information']).cuda();router.load_state_dict(checkpoint['router']);router.eval().requires_grad_(False)
        routers[name]=router
    versions=tuple(x._version for x in model.model.parameters());ds=ReplayDataset('val',require_complete=True)
    assert ds.fingerprint()==manifest['val_fingerprint'];pids=manifest['prompt_ids'][rank::world]
    modes=['Dense','mass','K200','V200','Kbest','Kpaused']
    all_rows={name:[]for name in modes};video_rows=[]
    try:
        # The original replay evaluator accesses tensor version counters; no_grad
        # preserves them. Evaluation never executes optimizer/backward calls.
        with torch.no_grad(),sdpa_backend():
            for name in modes:
                reset(model,name,routers,manifest)
                indices=[i for i,row in enumerate(ds.rows)if row['prompt_id']in pids]
                for n,index in enumerate(indices,1):
                    sample=ds.load(index);result=model.replay(sample,keep=.2,sparse=name!='Dense',checkpoint=False,collect_aux=False,recent_gradient=False)
                    target=sample['target'].cuda().float();pred=result['x0'].float();diag=result['diagnostics'].cpu().tolist()
                    energy=float(target.square().sum());sse=float((pred-target).square().sum())
                    row={k:sample[k]for k in('sample_id','trajectory_id','prompt_id','generation_seed','history_tokens','step','timestep')}
                    row.update(sse=sse,energy=energy,nmse=sse/(energy+1e-8),actual_history_keep=1. if name=='Dense'else diag[0]/max(diag[1]*sample['history_tokens'],1))
                    all_rows[name].append(row);del result,pred,target
                    status(status='offline',mode=name,offline_completed=n,offline_total=len(indices))
                atomic_json(work/f'{name}_offline.json',dict(rows=all_rows[name],summary=summarize(all_rows[name])))
            if manifest['videos']:
                import lpips
                from utils.wan_wrapper import WanVAEWrapper
                metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).cuda().eval().requires_grad_(False)
                vae=WanVAEWrapper().cuda().to(dtype=torch.bfloat16).eval().requires_grad_(False)
                assert abs(float(metric(torch.zeros(1,3,64,64,device='cuda'),torch.zeros(1,3,64,64,device='cuda'))))<1e-7
                for pid in pids:
                    for seed in manifest['generation_seeds']:
                        index=next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==seed)
                        sample=ds.load(index);embeds=sample['prompt_embeds'].cuda().to(dtype=torch.bfloat16)
                        case_dir=root/'videos'/f'{pid}.seed{seed}';case_dir.mkdir(parents=True,exist_ok=True)
                        latents={};pixels={};details={}
                        for name in modes:
                            reset(model,name,routers,manifest);status(status='rollout',mode=name,prompt_id=pid,generation_seed=seed)
                            counts=[];original_forward=model.forward_chunk
                            def measured(*a,**kw):
                                r=original_forward(*a,**kw)
                                if kw.get('sparse',True)and a[3]:
                                    diag=r['diagnostics'].detach().cpu().tolist();history=a[3][0].k.shape[1]
                                    counts.append(dict(history_tokens=history,step=kw['step'],retained_query_tokens=diag[0],routed_queries=diag[1],keep=diag[0]/max(diag[1]*history,1)))
                                return r
                            model.forward_chunk=measured
                            try:chunks,diagnostics=trajectory(model,embeds,pid,seed,name!='Dense')
                            finally:model.forward_chunk=original_forward
                            latents[name]=torch.cat(chunks,1)
                            if name!='Dense':
                                assert torch.equal(chunks[0],latents['Dense'][:,:3]),'First dense chunk mismatch'
                                assert len(counts)==12 and all(c['keep']<=.2+1e-9 for c in counts)
                            details[name]=dict(diagnostics=diagnostics,sparsity=counts)
                            model._eval_history=None;model._eval_sketch.clear()
                            pixels[name]=vae.decode_to_pixel(latents[name].cuda().to(dtype=torch.bfloat16),use_cache=False)[0].cpu()
                            vae.model.clear_cache();assert pixels[name].shape==(45,3,480,832)
                            if name!='Dense':
                                assert torch.equal(pixels[name][:9],pixels['Dense'][:9]),'First dense RGB chunk mismatch'
                                details[name]['quality']=pixel_metrics(pixels['Dense'],pixels[name],metric)
                            atomic_json(case_dir/f'{name}_metrics.json',details[name])
                        save(case_dir/'latents.pt',latents)
                        preview(case_dir/'paired_Dense_K200_V200.mp4',pixels,['Dense','K200','V200'])
                        preview(case_dir/'current_Dense_Kbest_Kpaused.mp4',pixels,['Dense','Kbest','Kpaused'])
                        row=dict(prompt_id=pid,generation_seed=seed,metrics=details,preview_paired=str(case_dir/'paired_Dense_K200_V200.mp4'),preview_current=str(case_dir/'current_Dense_Kbest_Kpaused.mp4'))
                        video_rows.append(row);atomic_json(work/'video_rows.json',video_rows)
                        status(status='rollout',completed=len(video_rows),video_total=len(pids)*2)
                        del pixels,latents,chunks;gc.collect();torch.cuda.empty_cache()
        assert tuple(x._version for x in model.model.parameters())==versions,'Frozen backbone changed'
        assert not any(p.requires_grad for p in model.parameters())
        for name,r in routers.items():assert tensor_digest(r.state_dict().items())==manifest['checkpoints'][name]['router_sha256']
        status(status='completed',backbone_versions_unchanged=True,routers_unchanged=True)
    except BaseException as error:status(status='failed',error=repr(error));raise


if __name__=='__main__':main()
