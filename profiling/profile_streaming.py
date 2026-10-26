"""Profile the native dense Self-Forcing streaming pipeline without editing its source.

Runtime adaptation: 720p token indexing; rolling KV storage with exactly H complete
history chunks plus the current chunk. Four native denoise calls and one dense
clean-cache write remain unchanged. All timings distinguish CUDA spans and kernel sums.
"""
from __future__ import annotations
import argparse, ast, contextlib, csv, gc, gzip, hashlib, inspect, json, math, os
from pathlib import Path
import statistics, sys, textwrap, time, types
from collections import defaultdict
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def csv_write(path,rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def sync():torch.cuda.synchronize()
def pair():return torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

_MISSING=object()

class Session:
    def __init__(self,mode,run,out):
        self.mode=mode;self.run=run;self.out=out
        self.ctx=dict(chunk=-1,step=-1,stage='setup',layer=-1)
        self.chunks=[];self.forwards=[];self.core=[];self.restores=[];self.samples={}
        self.chunk_index=-1;self.forward_index=0;self.kind='other'
    def label(self,part):
        c=self.ctx
        return f"SFPROF|c={c['chunk']}|s={c['step']}|p={c['stage']}|l={c['layer']}|part={part}"
    @contextlib.contextmanager
    def region(self,part):
        if self.mode=='detail':
            with torch.profiler.record_function(self.label(part)):yield
        else:yield
    @contextlib.contextmanager
    def chunk(self):
        self.chunk_index+=1;self.forward_index=0
        self.ctx.update(chunk=self.chunk_index,step=-1,stage='chunk',layer=-1)
        start,end=pair();torch.cuda.reset_peak_memory_stats();wall=time.perf_counter();start.record()
        with self.region('chunk.total'):yield
        end.record();end.synchronize()
        row=dict(run=self.run,chunk=self.chunk_index+1,chunk_zero_based=self.chunk_index,
            cuda_span_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        self.chunks.append(row)
        print(json.dumps(dict(progress='chunk_complete',mode=self.mode,**row)),flush=True)
    def replace(self,obj,name,value):
        self.restores.append((obj,name,obj.__dict__.get(name,_MISSING)));setattr(obj,name,value)
    def wrap(self,obj,part,layer=None):
        old=obj.forward
        def f(this,*args,**kwargs):
            previous=self.ctx['layer']
            if layer is not None:self.ctx['layer']=layer
            try:
                with self.region(part):return old(*args,**kwargs)
            finally:self.ctx['layer']=previous
        self.replace(obj,'forward',types.MethodType(f,obj))
    def install(self,pipe):
        import torch.nn.functional as F
        import wan.modules.causal_model as cm
        old=pipe.generator.forward
        def generator(this,*args,**kwargs):
            i=self.forward_index;self.forward_index+=1
            self.ctx.update(step=i if i<4 else -1,stage='denoise' if i<4 else 'context',layer=-1)
            a,b=pair();a.record()
            with self.region('model.total'):result=old(*args,**kwargs)
            b.record();self.forwards.append((dict(self.ctx),a,b))
            return result
        self.replace(pipe.generator,'forward',types.MethodType(generator,pipe.generator))
        if self.mode=='clean':return
        for idx,block in enumerate(pipe.generator.model.blocks):
            self.wrap(block,'block.total',idx)
        original_sdpa=F.scaled_dot_product_attention
        def sdpa(q,k,v,*args,**kwargs):
            kind='self' if k.shape[-2]!=512 else 'cross'
            metadata=dict(self.ctx,kind=kind,q_tokens=int(q.shape[-2]),kv_tokens=int(k.shape[-2]),heads=int(q.shape[1]),head_dim=int(q.shape[-1]))
            a,b=pair() if self.mode=='core' else (None,None)
            if a is not None:a.record()
            with self.region(kind+'.sdpa_fused'):result=original_sdpa(q,k,v,*args,**kwargs)
            if b is not None:
                b.record();self.core.append((metadata,a,b))
            if self.mode=='capture' and kind=='self' and self.ctx['layer']==14 and self.ctx['step']==3 and self.ctx['chunk']<4:
                self.samples[self.ctx['chunk']]={'q':q.cpu(),'k':k.cpu(),'v':v.cpu(),**metadata}
            return result
        self.replace(F,'scaled_dot_product_attention',sdpa)
        if self.mode!='detail':return
        self.replace(cm,'__sf_profile_region__',self.region) if hasattr(cm,'__sf_profile_region__') else setattr(cm,'__sf_profile_region__',self.region)
        # AST instrumentation adds scopes around unchanged original expressions.
        original_rope=cm.causal_rope_apply
        rope_calls=[0]
        def rope(*args,**kwargs):
            part='self.rope_q' if rope_calls[0]%2==0 else 'self.rope_k';rope_calls[0]+=1
            with self.region(part):return original_rope(*args,**kwargs)
        self.replace(cm,'causal_rope_apply',rope)
        self.replace(cm.CausalWanSelfAttention,'forward',instrument_self(cm.CausalWanSelfAttention.forward,cm.__dict__))
        self.replace(cm.CausalWanAttentionBlock,'forward',instrument_block(cm.CausalWanAttentionBlock.forward,cm.__dict__))
        # Existing instance wrappers bind original methods; update block wrappers to AST versions.
        for idx,block in enumerate(pipe.generator.model.blocks):
            block.forward=types.MethodType(cm.CausalWanAttentionBlock.forward,block)
            self.wrap(block,'block.total',idx)
            for name in ('norm1','norm2','norm3'):self.wrap(getattr(block,name),'block.'+name)
            for kind,obj in [('self',block.self_attn),('cross',block.cross_attn)]:
                self.wrap(obj,kind+'.total')
                for name in ('q','k','v','o','norm_q','norm_k'):
                    self.wrap(getattr(obj,name),kind+'.'+({'q':'q_projection','k':'k_projection','v':'v_projection','o':'output_projection','norm_q':'q_norm','norm_k':'k_norm'}[name]))
            self.wrap(block.ffn,'ffn.total')
            for name,obj in zip(('linear_in','gelu','linear_out'),block.ffn):self.wrap(obj,'ffn.'+name)
        model=pipe.generator.model
        for name in ('patch_embedding','time_embedding','time_projection','text_embedding','head'):
            self.wrap(getattr(model,name),'model.'+name,-1)
    def close(self):
        for obj,name,value in reversed(self.restores):
            if value is _MISSING:delattr(obj,name)
            else:setattr(obj,name,value)
        self.restores.clear()
    def rows(self):
        forward=[dict(run=self.run,**m,cuda_span_ms=a.elapsed_time(b)) for m,a,b in self.forwards]
        # Keep zero-based internal metadata explicit while public chunk is one-based.
        for r in forward:r['chunk_zero_based']=r['chunk'];r['chunk']+=1
        core=[]
        for m,a,b in self.core:
            r=dict(run=self.run,**m,cuda_span_ms=a.elapsed_time(b));r['chunk_zero_based']=r['chunk'];r['chunk']+=1;core.append(r)
        return forward,core


def scoped(statements,part):
    node=ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='__sf_profile_region__',ctx=ast.Load()),args=[ast.Constant(part)],keywords=[]))],body=statements)
    return ast.copy_location(node,statements[0])

def compile_method(old,tree,namespace):
    tree=ast.fix_missing_locations(tree);new=namespace.copy();exec(compile(tree,inspect.getsourcefile(old),'exec'),new)
    return new[old.__name__]

def instrument_self(old,namespace):
    tree=ast.parse(textwrap.dedent(inspect.getsource(old)));fn=tree.body[0]
    branch=next(x for x in fn.body if isinstance(x,ast.If) and ast.unparse(x.test)=='kv_cache is None')
    body=branch.orelse
    lo=next(i for i,x in enumerate(body) if isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='current_end')
    hi=next(i for i,x in enumerate(body) if isinstance(x,ast.Assign) and isinstance(x.value,ast.Call) and ast.unparse(x.value.func)=='attention')
    assert lo<hi
    branch.orelse=body[:lo]+[scoped(body[lo:hi],'self.kv_cache_update'),scoped([body[hi]],'self.attention_backend'),scoped(body[hi+1:],'self.cache_metadata')]
    return compile_method(old,tree,namespace)

def instrument_block(old,namespace):
    tree=ast.parse(textwrap.dedent(inspect.getsource(old)));fn=tree.body[0]
    out=[]
    for x in fn.body:
        if isinstance(x,ast.Assign):
            target=ast.unparse(x.targets[0]);source=ast.unparse(x.value)
            part=None
            if target=='e':part='block.modulation'
            elif target=='y' and 'self.self_attn(' in source:part='block.self_attention_branch'
            elif target=='x' and 'y.unflatten' in source:part='block.self_residual'
            if part:x=scoped([x],part)
        elif isinstance(x,ast.FunctionDef) and x.name=='cross_attn_ffn':
            b=[]
            for st in x.body:
                if isinstance(st,ast.Assign):
                    src=ast.unparse(st.value)
                    part='block.cross_attention_branch' if 'self.cross_attn(' in src else 'block.ffn_branch' if 'self.ffn(' in src else 'block.ffn_residual'
                    st=scoped([st],part)
                b.append(st)
            x.body=b
        out.append(x)
    fn.body=out
    return compile_method(old,tree,namespace)


def install_chunk_scope():
    import pipeline.causal_inference as p
    old=p.CausalInferencePipeline.inference
    tree=ast.parse(textwrap.dedent(inspect.getsource(old)));fn=tree.body[0]
    target=next(x for x in fn.body if isinstance(x,ast.For) and ast.unparse(x.target)=='current_num_frames')
    target.body=[ast.copy_location(ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_sf_profile_chunk',ctx=ast.Load()),args=[],keywords=[]))],body=target.body),target.body[0])]
    p.CausalInferencePipeline.inference=compile_method(old,tree,p.__dict__)

class CachedText(torch.nn.Module):
    def __init__(self,embeds):super().__init__();self.embeds=embeds
    def forward(self,text_prompts):return {'prompt_embeds':self.embeds}
class EmptyDecoder(torch.nn.Module):
    def decode_to_pixel(self,latent,use_cache=False):return latent.new_empty((0,))

@torch.inference_mode()
def validate_rolling(model,history_chunks,out):
    from streaming_sparse.v08.backbone import native_cache
    # Same native layer, actual 720p Q and RoPE positions, append-only vs compact KV.
    model.model.freqs=model.model.freqs.to('cuda')
    layer=model.model.blocks[14].self_attn
    cap=(history_chunks+1)*10800
    def cache(n):
        return dict(k=torch.zeros(1,n,12,128,device='cuda',dtype=torch.bfloat16),v=torch.zeros(1,n,12,128,device='cuda',dtype=torch.bfloat16),global_end_index=torch.zeros(1,device='cuda',dtype=torch.long),local_end_index=torch.zeros(1,device='cuda',dtype=torch.long))
    full=cache(27*10800);roll=cache(cap)
    old=layer.local_attn_size;previous_max=layer.max_attention_size;layer.max_attention_size=cap
    grid=torch.tensor([[3,45,80]]);seq=torch.tensor([10800]);rows=[]
    rng=torch.Generator(device='cuda').manual_seed(20261022)
    for c in range(27):
        # Two overwrites test both advancing and repeated diffusion/cache writes.
        for stage in range(2):
            x=torch.randn(1,10800,1536,device='cuda',dtype=torch.bfloat16,generator=rng)*.1
            args=(x,seq,grid,model.model.freqs,None)
            layer.local_attn_size=-1;a=layer(*args,kv_cache=full,current_start=c*10800)
            layer.local_attn_size=12;b=layer(*args,kv_cache=roll,current_start=c*10800)
            assert torch.equal(a,b),f'Rolling differs from append-only at chunk {c}, write {stage}'
        rows.append(dict(chunk=c+1,bitwise_equal=True,visible_tokens=min((c+1)*10800,cap)))
    layer.local_attn_size=old;layer.max_attention_size=previous_max
    del full,roll,a,b,x;torch.cuda.empty_cache()
    dump(out/'rolling_equivalence.json',dict(passed=True,comparison='same original layer14, actual 720p shapes, all27 chunks, 2 writes/chunk, append-only storage vs rolling storage, identical visible key window',rows=rows))


def parse_label(name):
    fields=dict(x.split('=',1) for x in name.split('|')[1:])
    return dict(chunk=int(fields['c'])+1,step=int(fields['s']),stage=fields['p'],layer=int(fields['l']),part=fields['part'])

def parse_profile(prof,out,trace):
    rows=defaultdict(lambda:[0.,0,0.]);operators=defaultdict(lambda:[0.,0]);inclusive=[]
    with gzip.open(out/'kernel_events.jsonl.gz','wt') as raw:
        for e in prof.events():
            if e.name.startswith('SFPROF|'):
                inclusive.append(dict(**parse_label(e.name),gpu_kernel_inclusive_ms=e.device_time_total/1000,cpu_span_ms=e.cpu_time_total/1000))
            if not e.kernels:continue
            parent=e
            while parent is not None and not parent.name.startswith('SFPROF|'):parent=parent.cpu_parent
            meta=parse_label(parent.name) if parent is not None else dict(chunk=0,step=-1,stage='outside',layer=-1,part='outside')
            key=tuple(meta[k] for k in ('chunk','step','stage','layer','part'))
            rows[key][2]+=e.self_cpu_time_total/1000
            for k in e.kernels:
                ms=k.duration/1000;rows[key][0]+=ms;rows[key][1]+=1
                opkey=(meta['part'],e.name,k.name);operators[opkey][0]+=ms;operators[opkey][1]+=1
                raw.write(json.dumps(dict(**meta,operator=e.name,kernel=k.name,device=k.device,gpu_ms=ms))+'\n')
    exclusive=[dict(zip(('chunk','step','stage','layer','part'),key),gpu_kernel_ms=v[0],kernel_count=v[1],cpu_self_ms_of_launching_ops=v[2]) for key,v in rows.items()]
    csv_write(out/'exclusive_kernels.csv',exclusive);csv_write(out/'inclusive_ranges.csv',inclusive)
    csv_write(out/'operators.csv',[dict(part=k[0],operator=k[1],kernel=k[2],gpu_ms=v[0],calls=v[1]) for k,v in sorted(operators.items(),key=lambda kv:-kv[1][0])])
    if trace:
        path=out/'trace.json';prof.export_chrome_trace(str(path))
        with path.open('rb') as src,gzip.open(out/'trace.json.gz','wb',compresslevel=3) as dst:
            import shutil;shutil.copyfileobj(src,dst)
        path.unlink()
    return dict(gpu_kernel_ms=sum(v[0] for v in rows.values()),kernel_count=sum(v[1] for v in rows.values()),attribution='Each directly correlated CUDA kernel assigned exactly once to deepest SFPROF scope. Inclusive ranges overlap and must not be summed.')


def microbenchmark(samples,out,repeats=3,tile=512):
    results=[]
    with torch.inference_mode():
        for c,sample in sorted(samples.items()):
            q,k,v=[sample[x].cuda() for x in ('q','k','v')]
            fused= torch.nn.functional.scaled_dot_product_attention(q,k,v)
            fused_ms=[]
            for _ in range(10):
                a,b=pair();a.record();result=torch.nn.functional.scaled_dot_product_attention(q,k,v);b.record();b.synchronize();fused_ms.append(a.elapsed_time(b))
            for rep in range(repeats+1):
                events=[];pieces=[]
                for start in range(0,q.shape[-2],tile):
                    qq=q[:,:,start:start+tile]
                    a,b=pair();a.record();scores=qq@k.transpose(-1,-2);b.record();events.append(('qk',a,b))
                    a,b=pair();a.record();scores.mul_(q.shape[-1]**-.5);b.record();events.append(('scale',a,b))
                    a,b=pair();a.record();prob=torch.softmax(scores,dim=-1,dtype=torch.float32);b.record();events.append(('softmax_fp32',a,b))
                    a,b=pair();a.record();prob_bf16=prob.to(q.dtype);b.record();events.append(('probability_cast',a,b))
                    a,b=pair();a.record();vv=prob_bf16@v;b.record();events.append(('pv',a,b));pieces.append(vv)
                    del scores,prob,prob_bf16
                sync();totals=defaultdict(float)
                for name,a,b in events:totals[name]+=a.elapsed_time(b)
                output=torch.cat(pieces,dim=-2)
                if rep:
                    diff=output.float()-fused.float()
                    results.append(dict(chunk=c+1,repeat=rep,q_tokens=q.shape[-2],kv_tokens=k.shape[-2],query_tile=tile,fused_full_query_ms_median=statistics.median(fused_ms),**{name+'_ms':value for name,value in totals.items()},explicit_total_ms=sum(totals.values()),max_abs_vs_fused=diff.abs().max().item(),relative_rmse_vs_fused=(diff.square().sum()/fused.float().square().sum()).sqrt().item(),measurement_scope='DIAGNOSTIC explicit tiled attention; these are NOT internal timings of the fused baseline kernel'))
                del output,pieces
            del q,k,v,fused,result;torch.cuda.empty_cache()
            print(json.dumps(dict(progress='attention_parts_complete',chunk=c+1)),flush=True)
    csv_write(out/'attention_parts_diagnostic.csv',results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--chunks',type=int,default=27);parser.add_argument('--history-chunks',type=int,default=3);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--skip-trace',action='store_true');parser.add_argument('--skip-diagnostic',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['HF_HUB_OFFLINE']='1'
    from omegaconf import OmegaConf
    from streaming_sparse.v08.common import source_identity
    from streaming_sparse.v08.backbone import load_teacher,native_steps
    from utils.wan_wrapper import WanTextEncoder,WanVAEWrapper
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from pipeline.causal_inference import CausalInferencePipeline
    import wan.modules.causal_model as cm
    status=dict(status='preparing',pid=os.getpid(),chunks=args.chunks,history_chunks=args.history_chunks)
    def publish(**kw):status.update(kw,updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat());dump(out/'status.json',status);print(json.dumps(status),flush=True)
    metadata=dict(request='Native dense Self-Forcing 1.3B streaming, real 1280x720, 27 chunks, at most3 history chunks, 4 native denoising steps plus one dense clean-cache write each chunk',seed=args.seed,resolution=[720,1280],latent_shape=[1,args.chunks*3,16,90,160],latent_frames_per_chunk=3,tokens_per_frame=3600,query_tokens=10800,max_history_tokens=args.history_chunks*10800,max_visible_kv_tokens=(args.history_chunks+1)*10800,latent_frames=args.chunks*3,decoded_frames=(args.chunks*3-1)*4+1,torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_properties=str(torch.cuda.get_device_properties(0)),dtype='bfloat16',source_sha256=source_identity(),git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend='native PyTorch SDPA (no external flash-attn), selected CUDA kernel recorded by profiler',tf32=False,torch_compile=False,window_includes_current=True,context_write='dense all30 layers, not skipped',sparse_model_used=False,timing_scope='Models loaded; text encoding, cache init, denoising, clean-cache writes, scheduler, VAE measured separately. Disk model loading and MP4 encode/write excluded.',runtime_adaptations=['frame_seq_length=3600 for true720p','local_attn_size=12 latent frames to enable native rolling cache','self_attention.max_attention_size=43200 for3 history+1 current chunk','cache capacity43200 tokens per layer, original rolling code reused','text embeddings cached and text encoder unloaded before transformer profiling; its time reported separately'])
    dump(out/'metadata.json',metadata)
    try:
        with torch.inference_mode():
            publish(status='encoding_text')
            registry=json.loads((ROOT/'streaming_sparse/v08/configs/prompt_registry.json').read_text())
            # Registry stores prompt text and splits; select an unsealed training prompt.
            prompts=registry.get('texts',{})
            if prompts:
                pid=registry['splits']['train'][0];prompt=prompts[pid]
            else:
                first=registry['prompts'][0];pid=first.get('id','profile');prompt=first.get('text',first.get('prompt'))
            metadata.update(prompt_id=pid,prompt=prompt)
            encoder=WanTextEncoder().to(device='cuda',dtype=torch.bfloat16).eval()
            # Three actual encodings; first cold and subsequent warm separated.
            enc=[]
            for i in range(3):
                sync();a,b=pair();wall=time.perf_counter();a.record();embeds=encoder([prompt])['prompt_embeds'];b.record();b.synchronize();enc.append(dict(repeat=i,cold=i==0,cuda_span_ms=a.elapsed_time(b),wall_ms=(time.perf_counter()-wall)*1000))
            torch.save(dict(prompt_id=pid,prompt=prompt,prompt_embeds=embeds.cpu()),out/'prompt_embedding.pt');csv_write(out/'text_encoding.csv',enc)
            del encoder;gc.collect();torch.cuda.empty_cache()
            publish(status='loading_teacher');generator=load_teacher('cuda');steps,_=native_steps(generator);metadata['native_timesteps']=steps
            generator.model.local_attn_size=(args.history_chunks+1)*3
            for block in generator.model.blocks:
                block.self_attn.local_attn_size=(args.history_chunks+1)*3
                block.self_attn.max_attention_size=(args.history_chunks+1)*10800
            config=OmegaConf.merge(OmegaConf.load(ROOT/'configs/default_config.yaml'),OmegaConf.load(ROOT/'configs/self_forcing_dmd.yaml'))
            pipe=CausalInferencePipeline(config,device=torch.device('cuda'),generator=generator,text_encoder=CachedText(embeds),vae=EmptyDecoder())
            pipe.frame_seq_length=3600;pipe.local_attn_size=(args.history_chunks+1)*3
            install_chunk_scope()
            metadata['cache_bytes']=30*(args.history_chunks+1)*10800*12*128*2*2
            dump(out/'metadata.json',metadata)
            with sdpa_backend():
                publish(status='validating_rolling_equivalence');validate_rolling(generator,args.history_chunks,out)
                def run(mode,index,n_chunks=args.chunks,profile=False):
                    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
                    # Allocate full requested noise even for short passes so RNG matches full video.
                    noise=torch.randn(1,args.chunks*3,16,90,160,device='cuda',dtype=torch.bfloat16)
                    pipe.kv_cache1=None;pipe.crossattn_cache=None;torch.cuda.empty_cache()
                    a,b=pair();a.record();init_wall=time.perf_counter();pipe._initialize_kv_cache(1,torch.bfloat16,'cuda');pipe._initialize_crossattn_cache(1,torch.bfloat16,'cuda');b.record();b.synchronize()
                    initialization=dict(cuda_span_ms=a.elapsed_time(b),wall_ms=(time.perf_counter()-init_wall)*1000)
                    session=Session(mode,index,out);pipe._sf_profile_chunk=session.chunk;session.install(pipe)
                    profiler=None
                    try:
                        if profile:
                            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
                            profiler.__enter__()
                        sync();wall=time.perf_counter()
                        _,latent=pipe.inference(noise[:,:n_chunks*3],text_prompts=[prompt],return_latents=True,profile=False,low_memory=False)
                        sync();elapsed=(time.perf_counter()-wall)*1000
                        if profiler:profiler.__exit__(None,None,None)
                        forward,core=session.rows()
                        rows=dict(mode=mode,repeat=index,total_wall_ms=elapsed,initialization=initialization,chunks=session.chunks,forwards=forward,core=core)
                        dump(out/f'{mode}_{index}.json',rows)
                        if profiler:
                            rows['kernel_summary']=parse_profile(profiler,out,not args.skip_trace);dump(out/f'{mode}_{index}.json',rows)
                        return latent,session,rows
                    finally:session.close()
                publish(status='warmup');latent,_,warm=run('clean',-1,min(args.history_chunks+1,args.chunks));del latent
                reference=None
                for i in range(args.repeats):
                    publish(status='baseline',repeat=i)
                    latent,_,rows=run('clean',i)
                    if reference is None:
                        reference=latent.cpu();torch.save(reference,out/'reference_latents.pt')
                    else:assert torch.equal(latent.cpu(),reference),'Repeated clean run differs'
                    del latent
                publish(status='attention_core_probe');latent,_,rows=run('core',0)
                assert torch.equal(latent.cpu(),reference),'Core probe changed model output';del latent
                publish(status='detailed_profiler');latent,_,rows=run('detail',0,profile=True)
                assert torch.equal(latent.cpu(),reference),'Detailed instrumentation changed output';del latent
                if not args.skip_diagnostic:
                    publish(status='capture_attention_inputs');latent,sample_session,_=run('capture',0,min(args.history_chunks+1,args.chunks));del latent
                    pipe.kv_cache1=None;pipe.crossattn_cache=None;torch.cuda.empty_cache()
                    publish(status='diagnostic_explicit_attention');microbenchmark(sample_session.samples,out)
                    sample_session.samples.clear()
            # Release generator and cache before separately timing actual streaming VAE.
            pipe.kv_cache1=None;pipe.crossattn_cache=None;del generator,pipe,embeds;gc.collect();torch.cuda.empty_cache()
            publish(status='streaming_vae_decode');vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
            latent=reference.cuda();decode_rows=[]
            # Warm up separate decoder, then clear its causal cache for the full sequence.
            tmp=vae.decode_to_pixel(latent[:,:3],use_cache=False);del tmp;vae.model.clear_cache();sync()
            for c in range(args.chunks):
                a,b=pair();sync();wall=time.perf_counter();a.record()
                pixels=vae.decode_to_pixel(latent[:,c*3:(c+1)*3],use_cache=True)
                b.record();b.synchronize()
                assert pixels.shape[-2:]==(720,1280) and torch.isfinite(pixels).all()
                decode_rows.append(dict(chunk=c+1,cuda_span_ms=a.elapsed_time(b),wall_ms=(time.perf_counter()-wall)*1000,rgb_frames=pixels.shape[1],height=pixels.shape[-2],width=pixels.shape[-1]))
                del pixels
            assert sum(x['rgb_frames'] for x in decode_rows)==metadata['decoded_frames']
            vae.model.clear_cache();csv_write(out/'vae_decode_chunks.csv',decode_rows)
            del vae,latent;gc.collect();torch.cuda.empty_cache()
            metadata['source_sha256_after']=source_identity();assert metadata['source_sha256_after']==metadata['source_sha256']
            dump(out/'metadata.json',metadata);publish(status='completed')
    except BaseException as error:
        publish(status='failed',error=repr(error));raise

if __name__=='__main__':main()
