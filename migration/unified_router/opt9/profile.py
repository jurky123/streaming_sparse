"""Post-benchmark CPU/CUDA profile for the optimized and shared-query adapters."""
import argparse,gzip,json,time
from pathlib import Path
from contextlib import nullcontext
import torch
from ..common import config,atomic_json,update,paired_seed
from .network import load as load_fast
import os

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--stride',type=int,default=1)
    p.add_argument('--wait-suite',action='store_true');p.add_argument('--variant',default='Skip2');a=p.parse_args();root,_=config(a.directory);base=root/'optimization9'
    if a.wait_suite:
        while True:
            states=[base/f'quality/rank{i}.json'for i in range(8)]
            if all(f.exists()for f in states):
                values=[json.loads(f.read_text())for f in states]
                if all(v['status']=='completed'for v in values):break
            time.sleep(5)
    step=int(os.environ.get('SLIMAV_PROFILE_STEP','3'));out=base/f'profile_{a.variant}_step{step}';out.mkdir(exist_ok=True);paired_seed(0);os.environ['SLIMAV_OPT9']=a.variant;mode='U-Sol'if a.variant.endswith('Fast')else'K32-Set'
    update(out/'status.json',status='loading')
    net,_=load_fast(root,graph=True,fused=True,history_cache=True,compiled_build=True,
        query_stride=1,group=32)
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.capture.capture_runner import sdpa_backend
    ds=ReplayDataset('val',require_complete=True);pid=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids'][0]
    sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0))
    embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
    if a.variant.startswith('Skip'):
        from .skip_stats import attention as audited_attention
        audited=[]
        def audit(layer,start,actual_step,q,k,v,sketch,prepared,mask,y):
            if audited or (layer,start,actual_step)!=(24,9,step):return
            yy,counts,executed=audited_attention(q,k,v,mask,warps=2,threshold=-2. if a.variant.startswith('Skip2')else 0.)
            nmse=float((yy.float()-y.float()).square().sum()/y.float().square().sum())
            assert nmse<1.e-6,nmse
            total=counts.numel()*mask.shape[-1];qk=int(counts.sum());pv=int(executed.sum())
            record=dict(scope='L24 chunk4 step'+str(step)+'; separate warmup instrumentation',qk_page_tiles=qk,pv_page_tiles=pv,dense_page_tiles=total,qk_tile_density=qk/total,pv_tile_density=pv/total,pv_tiles_skipped_fraction=1-pv/qk,instrumentation_vs_production_nmse=nmse)
            atomic_json(out/'skip_stats.json',record);audited.append(record)
        net.inspect_callback=audit
    with sdpa_backend():
        net.set_mode(mode);warm,_=trajectory(net,embeds,pid,0,True)
        reference=torch.cat(warm,1);del warm
        active=False;profiles=[];original_block=net._block
        def wrap(name,fn):
            def inner(*args,**kw):
                with (torch.profiler.record_function('OPT::'+name)if active else nullcontext()):return fn(*args,**kw)
            return inner
        net.make_part=wrap('current_prototypes',net.make_part)
        net.make_build=wrap('all_prototypes',net.make_build)
        net.prepare=wrap('prototype_projection',net.prepare)
        net.graph_route=wrap('route_graph_copy_and_replay',net.graph_route)
        from .. import fast_network
        attention=fast_network.sparse_attention
        fast_network.sparse_attention=wrap('sparse_attention_including_union',attention)
        block=net.model.blocks[24]
        for name,module in [('q_projection',block.self_attn.q),('k_projection',block.self_attn.k),('v_projection',block.self_attn.v),
                            ('output_projection',block.self_attn.o),('cross_attention',block.cross_attn),('ffn',block.ffn),
                            ('norm1',block.norm1),('norm2',block.norm2),('norm3',block.norm3)]:
            module.forward=wrap(name,module.forward)
        def observed(block,layer,x,e0,grid,freqs,context,hk,hv,start,sparse,step,*args,**kw):
            nonlocal active
            if (layer,start,sparse,step)!=(24,9,True,int(os.environ.get('SLIMAV_PROFILE_STEP','3'))):
                return original_block(block,layer,x,e0,grid,freqs,context,hk,hv,start,sparse,step,*args,**kw)
            torch.cuda.synchronize();active=True
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True)as prof:
                with torch.profiler.record_function('OPT::block_total'):
                    result=original_block(block,layer,x,e0,grid,freqs,context,hk,hv,start,sparse,step,*args,**kw)
            active=False;profiles.append(prof);return result
        net._block=observed;net.set_mode(mode);update(out/'status.json',status='profiling')
        chunks,_=trajectory(net,embeds,pid,0,True)
        assert torch.equal(torch.cat(chunks,1),reference)
        assert len(profiles)==1;prof=profiles[0]
        rows=[];ops=[]
        for event in prof.key_averages():
            if event.device_type!=torch.autograd.DeviceType.CPU:continue
            row=dict(name=event.key,count=event.count,cpu_ms=event.cpu_time_total/1000,
                cuda_ms=event.device_time_total/1000,self_cpu_ms=event.self_cpu_time_total/1000,
                self_cuda_ms=event.self_device_time_total/1000)
            (rows if event.key.startswith('OPT::')else ops).append(row)
        trace=out/'trace.json';prof.export_chrome_trace(str(trace))
        data=json.loads(trace.read_text());kernels={}
        for event in data['traceEvents']:
            if event.get('cat')=='kernel':
                entry=kernels.setdefault(event['name'],dict(name=event['name'],calls=0,milliseconds=0.))
                entry['calls']+=1;entry['milliseconds']+=event.get('dur',0)/1000
        with gzip.open(out/'trace.json.gz','wb')as f:f.write(trace.read_bytes())
        trace.unlink()
        atomic_json(out/'modules.json',dict(query_stride=1,variant=a.variant,rows=rows,operators=ops,
            kernels=sorted(kernels.values(),key=lambda r:-r['milliseconds']),
            actual_kernel_sum_ms=sum(r['milliseconds']for r in kernels.values()),
            observer_latents_bit_identical=True,scope=f'Layer24, fourth chunk, step{step}; separate profiled pass; CPU scopes only'))
    from ..optimization_report import attribution
    attribution(out)
    update(out/'status.json',status='completed')

if __name__=='__main__':main()
