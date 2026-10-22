"""Chunk quality from saved pixels; paired SDPA and warm noise-to-RGB timings."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
import types
import torch
from ..common import HERE,atomic_json,paired_seed,protocol,sha,source_identity
from ..backbone import FunctionalBackbone,load_teacher,native_steps
from ..cache import CacheIdentity,build_history
from ..evaluate import load_network
from ..execution import accelerate
from ..launcher.worker import require_execution_audit
from ..replay import ReplayDataset


def saved_quality(root):
    original=json.loads((root/'summary.json').read_text());videos=[]
    for item in original['results']:
        chunks=[]
        for chunk,(start,end)in enumerate(((0,9),(9,21),(21,33),(33,45))):
            rows=item['per_frame'][start:end];mse=sum(r['mse_rgb01']for r in rows)/len(rows)
            chunks.append(dict(chunk_zero_based=chunk,video_frames_one_based=[start+1,end],
                mse_rgb01=mse,psnr_db=-10*math.log10(mse)if mse else None,identical=mse==0,
                lpips=sum(r['lpips']for r in rows)/len(rows)))
        videos.append(dict(video_index=item['video_index'],prompt_id=item['prompt_id'],chunks=chunks))
    result=dict(checkpoint_sha256=original['request']['checkpoint_sha256'],videos=videos,
        definition='causal VAE: first3 latent frames decode to RGB frames1..9; each subsequent3 adds12 RGB frames; PSNR from pooled RGB MSE, LPIPS frame mean')
    atomic_json(root/'chunk_quality.json',result)
    steps=[]
    for item in original['results']:
        for step in range(4):
            rows=[r for r in item['sparsity']['events']if r['step']==step]
            history=sum(r['history_token_query_pairs']for r in rows)
            kept=sum(r['retained_history_token_query_pairs']for r in rows)
            steps.append(dict(video_index=item['video_index'],step=step,timestep=protocol()['native_timesteps'][step],
                history_sparsity=1-kept/history,all_layer_denoising_connection_sparsity=sum(r['dropped_token_query_pairs']for r in rows)/sum(r['all_layer_dense_token_query_pairs']for r in rows),
                chunks=[dict(chunk=r['chunk'],history_tokens=r['history_tokens'],history_sparsity=(1-r['retained_history_token_query_pairs']/r['history_token_query_pairs'])if r['history_token_query_pairs']else None)for r in rows]))
    atomic_json(root/'step_sparsity.json',dict(diffusion_steps_skipped=0,steps=steps,
        sparse_layers=[4,14,24],total_layers=30,heads='all12 in each sparse layer',
        context_write='Dense in every layer',current_segment='all visible KV kept',
        all_four_native_steps_execute=True))
    return original,result


@torch.inference_mode()
def generate(network,embeds,pid,seed,sparse):
    """Same native RNG calls/visibility as rollout.py, without CPU diagnostics."""
    p=protocol();device=next(network.model.parameters()).device;dtype=next(network.model.parameters()).dtype
    steps,_=native_steps(network.wrapper);rng=torch.Generator(device=device).manual_seed(seed)
    noise=torch.randn((1,12,*p['latent_shape']),generator=rng,device=device,dtype=dtype)
    transitions=torch.stack([torch.stack([torch.randn((3,*p['latent_shape']),generator=rng,device=device,dtype=dtype).unflatten(0,(1,3))for _ in range(3)])for _ in range(4)])
    history=noise[:,:0]
    network._eval_history=None;network._eval_sketch.clear()
    for chunk in range(4):
        identity=CacheIdentity(network.model_version,network.router_version,f'{pid}.seed{seed}',
            f'rollout.chunk{chunk}',0,chunk*4680,p['context_noise'],owner='student'if sparse else'teacher')
        cache=build_history(network,history,embeds,identity,recent_gradient=False,checkpoint=False)
        current=noise[:,chunk*3:(chunk+1)*3]
        for step,timestep in enumerate(steps):
            result=network.forward_chunk(current,embeds,timestep,cache,chunk*3,sparse=sparse,step=step,
                identity=identity,checkpoint=False,collect_aux=False)
            x0=result['x0']
            if step<3:
                current=network.wrapper.scheduler.add_noise(x0.flatten(0,1),transitions[chunk,step].flatten(0,1),
                    torch.full((3,),steps[step+1],device=device)).unflatten(0,(1,3))
        history=torch.cat((history,x0),1)
    return history


class AttentionProbe:
    def __init__(self,network,mode):
        self.network=network;self.mode=mode;self.context={};self.events=[];self.saved=[]
    def region(self,category,function,args,kwargs):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();result=function(*args,**kwargs);end.record()
        self.events.append((category,start,end,dict(self.context)))
        return result
    def __enter__(self):
        import torch.nn.functional as F
        from .. import attention
        self.sdpa=F.scaled_dot_product_attention;self.multi=attention._native_multi
        def sdpa(q,k,v,*args,**kwargs):
            if q.ndim==4 and q.shape[-2]==4680 and q.shape[-1]==128 and k.shape[-2]>=4680:
                return self.region('sdpa_self_attention',self.sdpa,(q,k,v,*args),kwargs)
            return self.sdpa(q,k,v,*args,**kwargs)
        def multi(*args,**kwargs):return self.region('masked_execution_including_mask_materialization',self.multi,args,kwargs)
        F.scaled_dot_product_attention=sdpa;attention._native_multi=multi
        forward=self.network.forward_chunk;block=self.network._block
        def forward_chunk(this,*args,**kwargs):
            timestep=args[2]if len(args)>2 else kwargs['timestep']
            start=args[4]if len(args)>4 else kwargs['start']
            identity=kwargs.get('identity')
            self.context.update(stage='context'if float(timestep)==0 else'denoise',
                source_chunk=start//3,target_chunk=(identity.history_end//4680)if identity is not None else start//3,
                step=kwargs.get('step',0),mode=self.mode)
            return forward(*args,**kwargs)
        def layer(this,*args,**kwargs):
            self.context['layer']=args[1]
            return block(*args,**kwargs)
        self.network.forward_chunk=types.MethodType(forward_chunk,self.network)
        self.network._block=types.MethodType(layer,self.network)
        self.saved=[(self.network,'forward_chunk',forward),(self.network,'_block',block)]
        if self.network.router is not None:
            for obj,name,category in((self.network,'_head','routing_score_and_budget'),(self.network.router,'build','historical_sketch')):
                old=getattr(obj,name)
                def timed(*args,_old=old,_category=category,**kwargs):return self.region(_category,_old,args,kwargs)
                setattr(obj,name,timed);self.saved.append((obj,name,old))
        return self
    def __exit__(self,*exc):
        import torch.nn.functional as F
        from .. import attention
        F.scaled_dot_product_attention=self.sdpa;attention._native_multi=self.multi
        for obj,name,old in self.saved:setattr(obj,name,old)
    def summary(self):
        rows=[];totals=defaultdict(float);counts=defaultdict(int)
        for category,start,end,context in self.events:
            ms=start.elapsed_time(end);row=dict(category=category,cuda_elapsed_ms=ms,**context);rows.append(row)
            key=category+'/'+context['stage'];totals[key]+=ms;counts[key]+=1
            if category=='sdpa_self_attention'and context['stage']=='denoise'and context.get('layer')in(4,14,24):
                totals['selected3layer_denoise_sdpa_ms']+=ms;counts['selected3layer_denoise_sdpa_ms']+=1
        assert counts['sdpa_self_attention/denoise']==480
        assert counts['sdpa_self_attention/context']==180
        assert counts['selected3layer_denoise_sdpa_ms']==48
        return dict(totals_ms=dict(totals),counts=dict(counts),events=rows,
            note='CUDA event spans on current stream; self-SDPA excludes router, mask construction and QKV/O projections. Native masked execution contains its SDPA, so categories overlap; do not sum all categories.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.input).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if not root.is_relative_to(HERE)or not out.is_relative_to(HERE):raise ValueError('Outputs must stay in v08')
    original,quality=saved_quality(root);request=original['request'];require_execution_audit();accelerate.install()
    state=dict(status='preparing',pid=os.getpid(),output=str(out),videos=2,completed=0,checkpoint_updates=request['checkpoint_updates'],
        checkpoint_sha256=request['checkpoint_sha256'],source_sha256=source_identity(),script_sha256=sha(Path(__file__)))
    def publish(status,**values):
        state.update(status=status,**values,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'status.json',state);atomic_json(HERE/'reports/two_video_evaluation.json',state)
        print(json.dumps(dict(status=status,**values),ensure_ascii=False),flush=True)
    paused=[]
    def interrupt(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    try:
        queue= json.loads((HERE/'reports/workflow/status.json').read_text())
        active=[v for v in queue['jobs'].values()if v['status']=='running'];assert len(active)==1
        scheduler=int((HERE/'reports/workflow/scheduler.pid').read_text())
        for pid in(scheduler,active[0]['pid']):
            if b'streaming_sparse.v08.launcher'not in Path(f'/proc/{pid}/cmdline').read_bytes():raise ValueError('Unrelated PID')
            os.kill(pid,signal.SIGSTOP);paused.append(pid)
        publish('running_timing_load',training_temporarily_suspended=True,paused_pids=paused)
        assert sha(root/'checkpoint.pt')==request['checkpoint_sha256']
        paired_seed(request['router_seed']);student,_=load_network(root/'checkpoint.pt','cuda')
        teacher=FunctionalBackbone(load_teacher('cuda'));teacher.model_version='frozen_teacher';teacher.router_version='none'
        from utils.wan_wrapper import WanVAEWrapper
        vae=WanVAEWrapper().eval().requires_grad_(False).to(device='cuda',dtype=torch.bfloat16)
        ds=ReplayDataset('val',require_complete=True)
        from ...capture.capture_runner import sdpa_backend
        results=[]
        # replay() validates storage version counters; load its warm-up input
        # before entering inference_mode, which intentionally removes counters.
        warm=ds.load(0)
        with torch.inference_mode(),sdpa_backend():
            for network,sparse in((teacher,False),(student,True)):
                network.replay(warm,sparse=sparse,checkpoint=False,collect_aux=False,recent_gradient=False)
                network._eval_history=None;network._eval_sketch.clear()
            saved=torch.load(root/'video1_latents.pt',weights_only=True,map_location='cpu')
            warm_pixels=vae.decode_to_pixel(saved['dense'][:,:3].to('cuda'),use_cache=False);del warm_pixels
            torch.cuda.synchronize();publish('running_timing',phase='warmup_complete')
            for video_index,prompt in enumerate(request['prompts'],1):
                pid=prompt['prompt_id'];sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0))
                embeds=sample['prompt_embeds'].to('cuda',dtype=torch.bfloat16)
                saved=torch.load(root/f'video{video_index}_latents.pt',weights_only=True,map_location='cpu')
                case=dict(video_index=video_index,prompt_id=pid,unprofiled={},profiled={})
                order=('dense','sparse')if video_index==1 else('sparse','dense')
                for mode in order:
                    network=teacher if mode=='dense'else student
                    publish('running_timing',video_index=video_index,mode=mode,phase='unprofiled_noise_to_rgb')
                    torch.cuda.synchronize();start=time.perf_counter()
                    latent=generate(network,embeds,pid,0,mode=='sparse');torch.cuda.synchronize();generation=time.perf_counter()-start
                    decode_start=time.perf_counter();pixels=vae.decode_to_pixel(latent,use_cache=False);torch.cuda.synchronize()
                    total=time.perf_counter()-start;decode=time.perf_counter()-decode_start
                    difference=float((latent.cpu().float()-saved[mode].float()).abs().max())
                    case['unprofiled'][mode]=dict(generation_wall_seconds=generation,vae_decode_wall_seconds=decode,
                        noise_to_rgb_wall_seconds=total,latent_max_abs_vs_original_video=difference,shape=list(pixels.shape))
                    del latent,pixels;vae.model.clear_cache()
                    atomic_json(out/f'video{video_index}.json',case)
                for mode in order:
                    network=teacher if mode=='dense'else student
                    publish('running_timing',video_index=video_index,mode=mode,phase='profiled_components')
                    with AttentionProbe(network,mode)as probe:
                        torch.cuda.synchronize();start=time.perf_counter();latent=generate(network,embeds,pid,0,mode=='sparse')
                        torch.cuda.synchronize();wall=time.perf_counter()-start
                    case['profiled'][mode]=dict(generation_wall_seconds=wall,**probe.summary());del latent
                    atomic_json(out/f'video{video_index}.json',case)
                case['noise_to_rgb_speedup_dense_over_sparse']=case['unprofiled']['dense']['noise_to_rgb_wall_seconds']/case['unprofiled']['sparse']['noise_to_rgb_wall_seconds']
                results.append(case);publish('running_timing',completed=len(results))
        summary=dict(checkpoint_updates=request['checkpoint_updates'],checkpoint_sha256=request['checkpoint_sha256'],results=results,
            mean_noise_to_rgb_dense_seconds=sum(r['unprofiled']['dense']['noise_to_rgb_wall_seconds']for r in results)/2,
            mean_noise_to_rgb_sparse_seconds=sum(r['unprofiled']['sparse']['noise_to_rgb_wall_seconds']for r in results)/2,
            timing_scope='Warm loaded models and cached text embeddings; all4 denoising steps/chunk, history reencoding, router and VAE decode to GPU RGB. Excludes model loading, text encoding, LPIPS, pixel transfer, video encoding/write. E2E timings use no component hooks; component timings come from separate repeats of the same2 cases, caches cleared between videos/repeats.',
            implementation='current v08 functional offline implementation; standard SDPA with hard mask, no sparse kernel; not original inference pipeline production throughput',
            training_source_sha256=source_identity(),execution_sha256=accelerate.identity())
        summary['mean_speedup_dense_over_sparse']=summary['mean_noise_to_rgb_dense_seconds']/summary['mean_noise_to_rgb_sparse_seconds']
        atomic_json(out/'summary.json',summary)
        lines=['Dense/Sparse warm noise-to-RGB timings; same F-K seed0 update620 checkpoint and same2 videos.',summary['timing_scope'],summary['implementation']]
        for case in results:
            lines.append(f'Video {case["video_index"]}')
            for mode in('dense','sparse'):
                e=case['unprofiled'][mode];p=case['profiled'][mode]['totals_ms']
                lines.append(f'  {mode}: generation {e["generation_wall_seconds"]:.6f}s; VAE {e["vae_decode_wall_seconds"]:.6f}s; total {e["noise_to_rgb_wall_seconds"]:.6f}s; denoise self-SDPA {p["sdpa_self_attention/denoise"]:.6f}ms; selected3layer self-SDPA {p["selected3layer_denoise_sdpa_ms"]:.6f}ms; routing {p.get("routing_score_and_budget/denoise",0):.6f}ms; sketch {p.get("historical_sketch/denoise",0):.6f}ms')
        (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');publish('completed',completed=2)
    except BaseException as error:publish('failed',error=repr(error));raise
    finally:
        resumed=[]
        for pid in reversed(paused):
            try:
                if b'streaming_sparse.v08.launcher'in Path(f'/proc/{pid}/cmdline').read_bytes():os.kill(pid,signal.SIGCONT);resumed.append(pid)
            except(FileNotFoundError,ProcessLookupError):pass
        state.update(training_temporarily_suspended=False,resumed_pids=resumed)
        atomic_json(out/'status.json',state);atomic_json(HERE/'reports/two_video_evaluation.json',state)
        print(json.dumps(dict(training_resumed=resumed)),flush=True)


if __name__=='__main__':main()
