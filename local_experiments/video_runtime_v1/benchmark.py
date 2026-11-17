"""Same-input microbenchmarks, original references versus runtime components."""
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
import torch
from .scores import grouped_scores
from .attention import Workspace,exact,compensate,merge
from .page import encode,read
from .weights import ROOT,load
from ..e30_beta_diagnostics.run import local_scores
from ..review_v1.ops import Summary
from ..review_v1.runtime import SelectedAttention
from ..learned_cache.io import atomic

def measure(fn,reps=7):
    for _ in range(3):fn()
    torch.cuda.synchronize();times=[];wall=[]
    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record();fn();b.record();b.synchronize()
        times.append(a.elapsed_time(b));wall.append((time.perf_counter()-start)*1000)
    return dict(gpu_ms_median=statistics.median(times),wall_ms_median=statistics.median(wall),gpu_ms=times,peak_extra_bytes=torch.cuda.max_memory_allocated()-base)

@torch.no_grad()
def run(output,module="scorer",event_batch=72):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[];identity=None
    if (out/'benchmark.json').exists():raise ValueError('Use a fresh output directory to preserve earlier measurements')
    if not 1<=event_batch<=864:raise ValueError('event_batch represents heads x tiles from one video, 1..864')
    if module in ('all','scorer'):
        z=np.load(ROOT/'artifacts/e49_numeric_v1/reference_input_17_05.npz')
        rp,b,lm,c,mask,r,a=[torch.as_tensor(z[n],device='cuda') for n in ('rp','beta','logmass','center','mask','r','a')]
        rp,b,lm,c=(v.double() for v in (rp,b,lm,c))
        nr=len(torch.unique_consecutive(r));na=len(r)//nr;rr=r.long().reshape(nr,na)[:,0];aa=a.long().reshape(nr,na)[0]
        for batch in (event_batch,):
            args=[v[None].expand(batch,*v.shape).contiguous() for v in (rp,b,lm,c,mask,rr,aa)]
            def old_score():
                return [local_scores(args[0][i],args[1][i],args[2][i],args[3][i],args[4][i],r.long(),a.long(),1.)[0] for i in range(batch)]
            old=measure(old_score,reps=5);new=measure(lambda:grouped_scores(*args),reps=5)
            rows.append(dict(component='512_swaps',batch=batch,baseline=old,optimized=new,speedup=old['wall_ms_median']/new['wall_ms_median'],scope='old local_scores includes its Independent diagnostic; Set-only reference separately below'))
            # Time the original Set-only loop without the unused Independent diagnostic.
            def old_set():
                result=[]
                for i in range(batch):
                    x,bb,ll,cc,mm=args[0][i],args[1][i],args[2][i],args[3][i],args[4][i];le=ll+bb
                    target=(le.softmax(1)[...,None]*(cc+x)).sum(1);logs=torch.where(mm[None],le,ll)
                    H=(logs.softmax(1)[...,None]*(cc+mm[None,:,None]*x)).sum(1);E=target-H;scores=[]
                    for st in range(0,len(r),32):
                        rs=r[st:st+32].long();ads=a[st:st+32].long();ms=mm[None].expand(len(rs),-1).clone();ix=torch.arange(len(rs),device=x.device);ms[ix,rs]=False;ms[ix,ads]=True
                        weights=torch.where(ms[None],le[:,None],ll[:,None]).softmax(-1)
                        Hnew=torch.bmm(weights,cc)+torch.bmm(weights*ms[None],x);delta=Hnew-H[:,None]
                        scores.append((2*(E[:,None]*delta).sum(-1)-delta.square().sum(-1)).mean(0))
                    result.append(torch.cat(scores))
                return result
            baseline=measure(old_set,reps=5);rows.append(dict(component='512_swaps_set_only',batch=batch,baseline=baseline,optimized=new,speedup=baseline['wall_ms_median']/new['wall_ms_median']))
    if module!='scorer':
        model,reader,phi,B,identity=load();h=12;p=410;t=6;n=t*64
        path=next((ROOT/'artifacts/full_coverage_v1/data/city_walk_s701').glob('c8_l15_s0_h0_t*.pt'));ev=torch.load(path,weights_only=True);kv=torch.load(path.parent/ev['kv_file'],weights_only=True)
        s=Summary(*(kv['summary'][name].cuda().expand(h,-1,*kv['summary'][name].shape[2:]).contiguous() for name in ('keys','mean','detail','stats')))
        p=s.mean.shape[1];q=ev['q'].cuda().bfloat16().repeat(t,1)[None].expand(h,-1,-1).contiguous();probes=q.reshape(h,t,64,128)[:,:,::4].contiguous()
        if module in ('all','readout'):
            def uncached():
                results=[]
                for tile in range(t):
                    co,A,_=encode(s,reader,phi,B.double());results.append(read(probes[:,tile:tile+1],co,A,phi))
                return results
            co,A,_=encode(s,reader,phi,B.double())
            old=measure(uncached);new=measure(lambda:read(probes,co,A,phi));enc=measure(lambda:encode(s,reader,phi,B.double()))
            rows.append(dict(component='page_readout',heads=h,tiles=t,pages=p,baseline=old,optimized=new,encode_once=enc,speedup=old['wall_ms_median']/new['wall_ms_median'],scope='baseline encodes every tile; optimized cache hit + batch readout; encode_once reported separately'))
        rawk=kv['k'].cuda()[None].expand(h,-1,-1,-1).contiguous();rawv=kv['v'].cuda()[None].expand_as(rawk).contiguous();basis=model[15].basis
        count=int(p*.1);mask=torch.zeros(h,t,p,device='cuda',dtype=torch.bool);mask[:,:,:count]=True
        ids=torch.arange(count,device='cuda',dtype=torch.int32)[None,None].expand(h,t,-1).contiguous();sel=dict(ids=ids,mask=mask,fixed_count=count)
        nq=q.transpose(0,1).contiguous()[None];nk=rawk.flatten(1,2).transpose(0,1).contiguous()[None];nv=rawv.flatten(1,2).transpose(0,1).contiguous()[None];work=Workspace()
        def fast_attention():
            logits=torch.bmm(q.float(),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,64,p,4)/128**.5
            oe,le=exact(nq,nk,nv,ids,work);oa,la=compensate(logits,s,basis,mask);return merge(oe,le,oa,la)
        if module in ('all','attention'):
            old=measure(lambda:SelectedAttention.fixed(q,rawk,rawv,s,basis,sel));new=measure(fast_attention)
            rows.append(dict(component='attention_full_mean_rank8',heads=h,tiles=t,pages=p,count=count,baseline=old,optimized=new,speedup=old['wall_ms_median']/new['wall_ms_median'],scope='includes Q64 summary QK in both; native KV resident before timing; no history repacking in optimized path'))
        if module in ('all','kv'):
            # Compare history concatenation with current-only writes on the same layout.
            hist=nk[:,:338*64];cur=nk[:,:72*64].clone();scratch=torch.empty(1,(338+72)*64,12,128,device='cuda',dtype=nk.dtype);scratch[:,:338*64].copy_(hist)
            old=measure(lambda:torch.cat((hist,cur),1));new=measure(lambda:scratch[:,338*64:].copy_(cur))
            rows.append(dict(component='KV_update_one_tensor',baseline=old,optimized=new,speedup=old['wall_ms_median']/new['wall_ms_median'],history_pages=338,current_pages=72))
    report=dict(rows=rows,identity=identity,video='city_walk_s701',module=module,device=torch.cuda.get_device_name(),torch=torch.__version__,scope='warmed component wall/CUDA-event medians; not an end-to-end video speedup',source={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    atomic(out/'benchmark.json',report)
    for row in rows:print(row['component'],row.get('batch',''),round(row['baseline']['wall_ms_median'],3),round(row['optimized']['wall_ms_median'],3),round(row['speedup'],2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--module',choices=['scorer','readout','attention','kv','all'],default='scorer');p.add_argument('--event-batch',type=int,default=72);a=p.parse_args();run(a.output or 'artifacts/video_runtime_v1/module_benchmark/'+a.module,a.module,a.event_batch)
