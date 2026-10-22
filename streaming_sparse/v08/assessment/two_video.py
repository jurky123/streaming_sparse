"""Exactly two paired Dense/Sparse videos: decoded RGB PSNR, LPIPS, sparsity."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import time
import types
import torch
from ..common import HERE,ROOT,atomic_json,paired_seed,protocol,save,sha,source_identity
from ..execution import accelerate
from ..launcher.worker import require_execution_audit
from ..backbone import FunctionalBackbone,load_teacher
from ..evaluate import load_network
from ..replay import ReplayDataset
from ..rollout import trajectory


def psnr(mse):return -10*math.log10(mse)if mse>0 else None


def sparsity(pair_rows,layers,heads,queries):
    grouped=defaultdict(list)
    for row in pair_rows:grouped[row['chunk'],row['step']].append(row)
    events=[]
    for chunk in range(4):
        for step in range(4):
            rows=grouped[chunk,step];history=chunk*4680
            assert len(rows)==(36 if chunk else 0)
            potential_history=36*queries*history
            kept=sum(r['retained_history_token_query_pairs']for r in rows)
            current=36*queries*queries
            skipped=potential_history-kept
            events.append(dict(chunk=chunk,step=step,history_tokens=history,
                history_token_query_pairs=potential_history,retained_history_token_query_pairs=kept,
                current_token_query_pairs=current,current_retained_in_full=True,
                all_layer_dense_token_query_pairs=layers*heads*queries*(history+queries),
                dropped_token_query_pairs=skipped))
    historical=sum(e['history_token_query_pairs']for e in events)
    kept=sum(e['retained_history_token_query_pairs']for e in events)
    current=sum(e['current_token_query_pairs']for e in events)
    all_dense=sum(e['all_layer_dense_token_query_pairs']for e in events)
    return dict(history_retained_ratio=kept/historical,history_sparsity=1-kept/historical,
        selected_layer_total_connection_sparsity=(historical-kept)/(historical+current),
        all_layer_denoising_connection_sparsity=(historical-kept)/all_dense,
        scope='logical self-attention Q-K connections during denoising; excludes dense context-write forwards, cross-attention and FFN; not measured kernel FLOPs',
        sparse_layers=[4,14,24],heads_per_layer=heads,total_backbone_layers=layers,
        current_retained_in_full=True,events=events)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();directory=Path(args.output).resolve()
    if not directory.is_relative_to(HERE):raise ValueError('Output must stay in v08')
    directory.mkdir(parents=True,exist_ok=True)
    require_execution_audit();accelerate.install()
    state=dict(status='preparing',pid=os.getpid(),output=str(directory),videos=2,completed=0,
               started_at=datetime.now(timezone.utc).isoformat(),source_sha256=source_identity(),
               execution_sha256=accelerate.identity(),assessment_sha256=sha(Path(__file__)))
    def publish(status,**values):
        state.update(status=status,**values,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(directory/'status.json',state)
        atomic_json(HERE/'reports/two_video_evaluation.json',state)
        print(json.dumps(dict(status=status,**values),ensure_ascii=False),flush=True)
    paused=[]
    def interrupt(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    try:
        # Load the exact pretrained metric before pausing any ongoing work.
        import lpips
        metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).eval().requires_grad_(False)
        workflow=HERE/'reports/workflow';queue=json.loads((workflow/'status.json').read_text())
        running=[(name,job)for name,job in queue['jobs'].items()if job['status']=='running']
        if len(running)!=1 or not running[0][0].startswith('main_'):
            raise RuntimeError('Expected exactly one active formal training job')
        name,job=running[0];_,group,seed_name=name.split('_');seed=int(seed_name.removeprefix('seed'))
        scheduler=int((workflow/'scheduler.pid').read_text())
        for pid in(scheduler,job['pid']):
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'streaming_sparse.v08.launcher'not in cmd:raise ValueError('Refuse to suspend an unrelated process')
            os.kill(pid,signal.SIGSTOP);paused.append(pid)
        publish('running_freeze_checkpoint',training_temporarily_suspended=True,paused_pids=paused)
        candidates=[]
        for tag in('recovery','latest','last','best'):
            path=HERE/f'runs/seed{seed}/{group}/{tag}.pt'
            if not path.exists():continue
            data=path.read_bytes();value=torch.load(io.BytesIO(data),weights_only=True,map_location='cpu')
            if value['source_sha256']!=source_identity()or value['pilot']or value['stage']!='main':raise ValueError('Wrong checkpoint identity')
            candidates.append((value['updates'],path.stat().st_mtime,path,data,value))
        updates,_,original,data,checkpoint=max(candidates,key=lambda item:(item[0],item[1]))
        frozen=directory/'checkpoint.pt';frozen.write_bytes(data)
        dataset=ReplayDataset('val',require_complete=True)
        prompts=sorted(dataset.registry['splits']['val'])[:2]
        request=dict(checkpoint_original_path=str(original),checkpoint_sha256=hashlib.sha256(data).hexdigest(),
            checkpoint_updates=updates,group=group,router_seed=seed,generation_seed=0,
            prompts=[dict(prompt_id=p,text=dataset.registry['texts'][p])for p in prompts],
            history_keep_target=.2,latent_chunks=4,latent_frames_per_chunk=3,
            psnr='RGB float [0,1], peak=1, global frame/pixel MSE, before MP4 compression',
            lpips='lpips0.1.4 AlexNet v0.1 pretrained, full-resolution frames in [-1,1], frame arithmetic mean',
            vae_dtype='BF16 as original inference pipeline',
            lpips_calibration_sha256=sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth'),
            alexnet_backbone_sha256=sha(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'),
            test_used=False,keep_current_kv=True,single_physical_gpu=0)
        atomic_json(directory/'request.json',request)
        publish('running_generation',checkpoint_updates=updates,checkpoint_group=group)
        paired_seed(seed);student,_=load_network(frozen,'cuda')
        teacher=FunctionalBackbone(load_teacher('cuda'));teacher.model_version='frozen_teacher';teacher.router_version='none'
        layers=len(student.model.blocks);heads=protocol()['heads'];cases=[]
        from ...capture.capture_runner import sdpa_backend
        with torch.inference_mode(),sdpa_backend():
            for video_index,pid in enumerate(prompts,1):
                publish('running_generation',video_index=video_index,prompt_id=pid,phase='dense')
                index=next(i for i,r in enumerate(dataset.rows)if r['prompt_id']==pid and r['generation_seed']==0)
                sample=dataset.load(index);embeds=sample['prompt_embeds'].to('cuda',dtype=torch.bfloat16)
                dense_chunks,dense_diagnostics=trajectory(teacher,embeds,pid,0,False)
                pair_rows=[];original_head=student._head
                def measured_head(self,*head_args,**kwargs):
                    result=original_head(*head_args,**kwargs)
                    q,kh=head_args[:2];sketch=head_args[5];keep=head_args[7]
                    retained=(result[0].detach()*sketch['costs']).sum(-1).long()
                    budget=math.floor(keep*len(kh));minimum=int(retained.min());maximum=int(retained.max())
                    if maximum>budget:raise AssertionError('History budget exceeded')
                    pair_rows.append(dict(chunk=len(kh)//4680,step=head_args[8],layer=head_args[14],head=head_args[15],
                        raw_queries=len(q),history_tokens=len(kh),budget=budget,
                        retained_min=minimum,retained_max=maximum,retained_mean=float(retained.float().mean()),
                        retained_history_token_query_pairs=int(retained.sum()),history_retained_ratio=float(retained.double().mean()/len(kh))))
                    return result
                student._head=types.MethodType(measured_head,student)
                publish('running_generation',video_index=video_index,phase='sparse')
                sparse_chunks,sparse_diagnostics=trajectory(student,embeds,pid,0,True)
                student._head=original_head
                assert len(pair_rows)==432
                dense_latent=torch.cat(dense_chunks,1);sparse_latent=torch.cat(sparse_chunks,1)
                save(directory/f'video{video_index}_latents.pt',dict(dense=dense_latent,sparse=sparse_latent))
                stats=sparsity(pair_rows,layers,heads,4680)
                atomic_json(directory/f'video{video_index}_sparsity.json',dict(summary=stats,per_pair=pair_rows))
                cases.append(dict(video_index=video_index,prompt_id=pid,generation_seed=0,sparsity=stats,
                    dense_diagnostics=dense_diagnostics,sparse_diagnostics=sparse_diagnostics))
                atomic_json(directory/'generation_summary.json',dict(request=request,cases=cases))
        del student,teacher,dataset,sample,embeds,original_head,measured_head
        gc.collect();torch.cuda.empty_cache()
        from utils.wan_wrapper import WanVAEWrapper
        publish('running_decode',phase='load_vae')
        vae=WanVAEWrapper().eval().requires_grad_(False).to(device='cuda',dtype=torch.bfloat16)
        metric=metric.to('cuda');results=[]
        import numpy as np
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        with torch.inference_mode():
            for case in cases:
                video_index=case['video_index'];latent=torch.load(directory/f'video{video_index}_latents.pt',weights_only=True,map_location='cpu')
                pixels={}
                for kind in('dense','sparse'):
                    publish('running_decode',video_index=video_index,phase=kind)
                    pixels[kind]=vae.decode_to_pixel(latent[kind].to('cuda',dtype=torch.bfloat16),use_cache=False)[0].cpu()
                    vae.model.clear_cache();torch.cuda.empty_cache()
                dense=pixels['dense'];sparse=pixels['sparse']
                assert dense.shape==sparse.shape and dense.shape[0]==45 and dense.shape[-2:]==(480,832)
                if not torch.isfinite(dense).all()or not torch.isfinite(sparse).all():raise FloatingPointError('Invalid decoded RGB')
                identity_lpips=float(metric(dense[:1].to('cuda'),dense[:1].to('cuda')).item())
                assert abs(identity_lpips)<1e-7
                publish('running_metrics',video_index=video_index,frames=len(dense))
                frame_rows=[];sse=0;count=0
                preview=directory/f'video{video_index}_dense_left_sparse_right.mp4'
                with imageio.get_writer(preview,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18'])as writer:
                    for frame in range(len(dense)):
                        a=dense[frame];b=sparse[frame]
                        # [-1,1] -> [0,1] halves the differences; use FP64 SSE.
                        error=(a.double()-b.double())*.5;energy=float(error.square().sum());n=error.numel()
                        value=float(metric(a[None].to('cuda'),b[None].to('cuda')).item())
                        if not math.isfinite(value):raise FloatingPointError('Invalid LPIPS')
                        frame_rows.append(dict(frame=frame,mse_rgb01=energy/n,psnr_db=psnr(energy/n),identical=energy==0,lpips=value))
                        sse+=energy;count+=n
                        left=((a.permute(1,2,0)+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
                        right=((b.permute(1,2,0)+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
                        image=Image.fromarray(np.concatenate((left,right),axis=1));draw=ImageDraw.Draw(image)
                        draw.text((8,8),'DENSE TEACHER',fill='white',stroke_width=2,stroke_fill='black')
                        draw.text((840,8),f'SPARSE {group} UPDATE {updates}',fill='white',stroke_width=2,stroke_fill='black')
                        writer.append_data(np.asarray(image))
                result={**case,**dict(frames=len(dense),height=480,width=832,fps=16,psnr_db=psnr(sse/count),
                    mse_rgb01=sse/count,lpips=sum(x['lpips']for x in frame_rows)/len(frame_rows),
                    per_frame=frame_rows,preview=str(preview),lpips_identity_check=identity_lpips)}
                results.append(result);atomic_json(directory/f'video{video_index}_metrics.json',result)
                publish('running_metrics',video_index=video_index,completed=len(results),psnr_db=result['psnr_db'],lpips=result['lpips'])
        summary=dict(request=request,results=results,mean_video_psnr_db=sum(r['psnr_db']for r in results)/2,
            mean_video_lpips=sum(r['lpips']for r in results)/2,
            note='Two fixed Val videos only; similarity to paired Dense Teacher, not ground truth video quality. Scores use uncompressed decoded pixels; previews are compressed. Full video includes initial dense chunk.')
        atomic_json(directory/'summary.json',summary)
        lines=[f'Two-video end-to-end check | {group} seed{seed} update{updates}',
            'Dense Teacher vs Sparse, same prompts/seed0/noise, four chunks = 12 latent frames = 45 RGB frames, 832x480, 16fps.',
            'PSNR: uncompressed decoded RGB [0,1], peak1. LPIPS: AlexNet v0.1, per-frame mean, no resize.',
            'Video | PSNR dB | LPIPS | history sparsity | all-layer denoising logical-connection sparsity']
        for r in results:lines.append(f'{r["video_index"]} | {r["psnr_db"]:.6f} | {r["lpips"]:.6f} | {r["sparsity"]["history_sparsity"]:.6%} | {r["sparsity"]["all_layer_denoising_connection_sparsity"]:.6%}')
        lines+=['Logical connection sparsity is not measured kernel FLOP/speed reduction; current KV kept, other27layers dense.',summary['note']]
        (directory/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
        publish('completed',completed=2)
    except BaseException as error:
        publish('failed',error=repr(error));raise
    finally:
        resumed=[]
        for pid in reversed(paused):
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                if b'streaming_sparse.v08.launcher'in cmd:os.kill(pid,signal.SIGCONT);resumed.append(pid)
            except ProcessLookupError:pass
            except FileNotFoundError:pass
        state.update(training_temporarily_suspended=False,resumed_pids=resumed,
                     finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(directory/'status.json',state);atomic_json(HERE/'reports/two_video_evaluation.json',state)
        print(json.dumps(dict(training_resumed=resumed)),flush=True)


if __name__=='__main__':main()
