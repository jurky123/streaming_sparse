import inspect,json,sys,time,textwrap,types
from pathlib import Path
import torch
here=Path(__file__).resolve().parent
repo=here.parents[1]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from light import build
from local_experiments.broad_common import noise
from local_experiments.anchor_rebase import shift_temporal_

def patch(fn,edits):
    src=textwrap.dedent(inspect.getsource(fn))
    for old,new in edits:
        assert src.count(old)==1,old
        if '\n' in new:
            indent=src[:src.index(old)].split('\n')[-1]
            parts=new.splitlines()
            shift=len(parts[-1])-len(parts[-1].lstrip())-len(indent)
            new='\n'.join([parts[0]]+[line[shift:] if shift>=0 else ' '*(-shift)+line for line in parts[1:]])
        src=src.replace(old,new)
    scope=dict(fn.__globals__,shift_temporal_=shift_temporal_)
    exec(compile(src,fn.__code__.co_filename,'exec'),scope)
    return scope[fn.__name__]

def install(pipe,method):
    blocks=pipe.generator.model.blocks
    if method=='forcingkv':
        att=type(blocks[0].self_attn)
        att.forward=patch(att.forward,[('switch_applied = bool', '''if cur_AR_step==0:
            history_keep_tokens=max(history_keep_tokens,2*frame_seqlen)
        if cur_AR_step>=6 and kv_cache.get("anchor_chunk")!=cur_AR_step:
            for key in ("sink_k","group_sink_spatial_k","group_sink_temporal_k"):
                if key in kv_cache:
                    dest=kv_cache[key]
                    assert dest.size(1)==frame_seqlen
                    original=kv_cache.setdefault("original_"+key,dest.clone()) if cur_AR_step==6 else kv_cache["original_"+key]
                    dest.copy_(original)
                    shift_temporal_(dest,frame_seqlen,freqs,current_start_frame-15)
            kv_cache["anchor_chunk"]=cur_AR_step
        switch_applied = bool''')])
        return
    cls=type(pipe)
    cls.inference=patch(cls.inference,[('if chunk_index > 0:','if chunk_index >= 2:')])
    att=type(blocks[0].self_attn)
    att.forward=patch(att.forward,[('kv_cache_size = kv_cache["k"].shape[1]','''if kv_cache.get("anchor_start")!=current_start_frame:
            if current_start_frame>=18:
                sizes=kv_cache["frame_sizes"]
                cutoff=current_start_frame-14
                expired=[f for f in sizes if 0<f<cutoff]
                drop=sum(sizes.pop(f) for f in expired)
                end=int(kv_cache["local_end_index"])
                if drop:
                    for field in ("k","v"):
                        data=kv_cache[field]
                        data[:,frame_seqlen:end-drop].copy_(data[:,frame_seqlen+drop:end].clone())
                    kv_cache["local_end_index"].sub_(drop)
                kv_cache["k"][:,:frame_seqlen].copy_(kv_cache["original_anchor"])
                shift_temporal_(kv_cache["k"],frame_seqlen,freqs,current_start_frame-15)
            kv_cache["anchor_start"]=current_start_frame
        kv_cache_size = kv_cache["k"].shape[1]'''),('k_flat = k_combined.flatten','''if current_start_frame<6:
                final_mask=torch.ones_like(final_mask)
            sizes=kv_cache.setdefault("frame_sizes",{})
            first=current_start_frame+num_frame-k_combined.shape[1]
            retained=final_mask.reshape(k_combined.shape[1],-1).sum(1).tolist()
            sizes.update((first+i,count) for i,count in enumerate(retained))
            assert sizes[0]==frame_seqlen
            if current_start_frame==0:
                kv_cache["original_anchor"]=k_view[:,0].clone()
            k_flat = k_combined.flatten''')])

@torch.inference_mode()
def run(method,n,smoke):
    torch.set_num_threads(2)
    pipe=build(n,window=18,method=method)
    install(pipe,method)
    from local_experiments.video_runtime_v1.ffn_rounded_gelu import RoundedFFN
    from local_experiments.video_runtime_v1.rms_dataflow import RMSDataflow
    from local_experiments.video_runtime_v1.rms_mean import row_fused
    engine=types.SimpleNamespace(wrapper=pipe.generator)
    RoundedFFN(engine)
    rms=RMSDataflow(engine,mean_kernel=True)
    rms.function=row_fused
    state={}
    checks={'early':0,'partial_or_grouped':0}
    def location(module,args,kw):
        state['c']=int(kw['current_start'])//4608
    hook=pipe.generator.register_forward_pre_hook(location,with_kwargs=True)
    scope=type(pipe.generator.model.blocks[0].self_attn).forward.__globals__
    attention=scope['attention']
    def audit(q,k,v,*args,**kwargs):
        c=state['c']
        if c<2:
            assert q.size(1)==4608 and k.size(1)==(c+1)*4608 and q.size(2)==12,(c,q.shape,k.shape)
            checks['early']+=1
        elif q.size(1)<4608 or q.size(2)<12:checks['partial_or_grouped']+=1
        return attention(q,k,v,*args,**kwargs)
    scope['attention']=audit
    pipe.text_encoder.embeds=torch.load(repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt',weights_only=True).cuda()
    init=torch.cat([noise(701,c,0) for c in range(n)],1)
    phases=[noise(701,c,p).flatten(0,1) for c in range(n) for p in (1,2,3)]
    times=[]
    for rep in range(1 if smoke else 3):
        pipe.kv_cache1=None
        for block in pipe.generator.model.blocks:
            a=block.self_attn
            for key in ('last_reference_frame','last_update_step','last_chunk_clean_k','last_chunk_clean_v','patch_dynamic_score'):
                if hasattr(a,key):setattr(a,key,None)
            if hasattr(a,'current_frame_num'):a.current_frame_num=0
        if hasattr(pipe.generator.model,'x_cache'):pipe.generator.model.x_cache.clear()
        stream=iter(phases);rand=torch.randn_like;torch.randn_like=lambda *a,**kw:next(stream)
        torch.cuda.synchronize();t=time.monotonic()
        try:_,z=pipe.inference(init,['cached'],return_latents=True,profile=False)
        finally:torch.randn_like=rand
        torch.cuda.synchronize();times.append(time.monotonic()-t)
        assert next(stream,None) is None and z.isfinite().all()
        for cache in pipe.kv_cache1:
            assert (cache['anchor_chunk']==n-1 if method=='forcingkv' else cache['anchor_start']==3*(n-1))
        if rep==0:
            hook.remove();scope['attention']=attention
            assert checks['early']==300 and checks['partial_or_grouped']>0,checks
        print(n,times[-1],checks,flush=True)
    if not smoke:
        torch.save(z.cpu(),here/f'{method}_{n}.pt')
        (here/f'{method}_{n}.json').write_text(json.dumps(dict(times=times,checks=checks)))
run(sys.argv[1],int(sys.argv[2]),'--smoke' in sys.argv)
