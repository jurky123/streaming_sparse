"""Steady-state real model torch.profiler trace plus separately measured stages."""
import argparse,gc,gzip,json,time,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .rollout_suite import generate_chunk
from .rollout_runtime import Runtime
from .gap_fast_runtime import GapFastRuntime
from .gap_staged_single_runtime import GapStagedSingleRuntime as GapFeatureRuntime
from .gap_tail_single import compensate as _compensate
from functools import partial
compensate=partial(_compensate,mode=1)
from .learned_cache.io import atomic
from .gap_peer_watch import PeerWatch
ROOT=Path('artifacts/gap_staged_single_20261101/profile')
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['dense','sparse','features'],required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(713);ROOT.mkdir(exist_ok=True,parents=True)
    from .gap_group_isolated import device_peers
    from .learned_cache.io import disk_guard
    assert not device_peers()
    disk_guard(ROOT,50,1)
    engine=Engine();emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
    rt=Runtime(engine,'dense') if args.variant=='dense' else (GapFeatureRuntime if args.variant=='features' else GapFastRuntime)(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
    if args.variant!='dense':rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
    try:
        for c in range(6):generate_chunk(engine,emb,101,c)
        events=[]
        def wrap(name,fn):
            def call(*a,**kw):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                with torch.profiler.record_function(name):value=fn(*a,**kw)
                end.record();events.append((name,start,end));return value
            return call
        if args.variant!='dense':
            rt.decision=wrap('sparse/route',rt.decision);rt.current_summary=wrap('sparse/current_summary',rt.current_summary);rt.commit=wrap('sparse/history_commit',rt.commit)
            import local_experiments.gap_fast_runtime as mod
            if args.variant=='features':
                import local_experiments.gap_flat_runtime as mod
            original=mod.flat_native
            def native(q,k,v,pk,base,detail,basis,center,scale,lut,kernel):
                _,n,h,d=q.shape;out=torch.empty_like(q);lse=torch.empty(1,h,n,device=q.device)
                def exact():kernel[(n//64,h)](q,k,v,d**-.5,lut.shape[-1],lut,lse,out,h,n,k.shape[1],n//64,d,64,64,num_warps=4,num_stages=3)
                wrap('sparse/exact',exact)();wrap('sparse/compensation',compensate)(q,pk,base,detail,lut,lut,out,lse,basis,center,torch.ones(h,device=q.device),scale)
                return out
            mod.flat_native=native
        torch.cuda.synchronize();start=time.perf_counter()
        with PeerWatch() as watch, torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
            generate_chunk(engine,emb,101,6)
        atomic(ROOT/'peer_samples.json',watch.summary());watch.validate()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        groups={}
        for name,a,b in events:
            row=groups.setdefault(name,dict(calls=0,total_ms=0.));row['calls']+=1;row['total_ms']+=a.elapsed_time(b)
        for row in groups.values():row['mean_ms']=row['total_ms']/row['calls']
        ops=[]
        for e in prof.key_averages():
            ops.append(dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,device_total_us=e.device_time_total,self_device_us=e.self_device_time_total))
        ops.sort(key=lambda x:x['self_device_us'],reverse=True)
        path=ROOT/(args.variant+'.trace.json');prof.export_chrome_trace(str(path))
        trace=json.loads(path.read_text());kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'];runtime=[e for e in trace['traceEvents'] if e.get('cat')=='cuda_runtime']
        bykernel={}
        for e in kernels:
            x=bykernel.setdefault(e['name'],dict(calls=0,total_us=0.));x['calls']+=1;x['total_us']+=e['dur']
        with path.open('rb') as src,gzip.open(str(path)+'.gz','wb') as dst:dst.write(src.read())
        path.unlink()
        result=dict(variant=args.variant,profiled_chunk_wall_seconds=elapsed,cuda_event_stages=groups,operators=ops,kernels=sorted([dict(name=k,**v) for k,v in bykernel.items()],key=lambda x:x['total_us'],reverse=True),kernel_count=len(kernels),kernel_sum_us=sum(e['dur'] for e in kernels),cuda_runtime_calls=len(runtime),note='Real chunk6 after six warm chunks; 4 denoise + clean, no VAE. Profiler perturbs wall time. CUDA events include queue gaps; kernel durations reported separately; no overlapping-stage sum used as wall time.')
        assert not device_peers()
        atomic(ROOT/(args.variant+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ['operators','kernels']}),flush=True)
    finally:
        if args.variant!='dense' and 'original' in locals():mod.flat_native=original
        rt.close()
if __name__=='__main__':main()
