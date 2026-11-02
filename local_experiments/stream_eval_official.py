"""Run an upstream pipeline in its own import namespace, with identical inputs.

No custom anchor rebasing, policy replacement, or DASA component ablation.
Only paths, attention backend compatibility, frame-token sizing, and budgets
are configured. Decoding is deferred to a common VAE in a separate process.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def storage_bytes(values):
    stores={}
    for x in values:
        if isinstance(x,torch.Tensor) and x.numel():
            s=x.untyped_storage();stores[s.data_ptr()]=s.nbytes()
    return sum(stores.values())


class SavedText(torch.nn.Module):
    def __init__(self,embeds):
        super().__init__();self.embeds=embeds
    def forward(self,text_prompts):
        return {'prompt_embeds':self.embeds}


class DeferredVAE(torch.nn.Module):
    def decode_to_pixel(self,latent,**kwargs):
        return latent.new_zeros((1,1,3,1,1))


def module_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['lightforcing','forcingkv','dasa','dense'],required=True)
    p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--chunks',type=int,default=41)
    p.add_argument('--research-config',help='Optional local research hook; never used by official baseline runs')
    a=p.parse_args()
    inputs_path=Path(a.inputs).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists(), 'Completed method must be skipped by controller'
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs={k:v.cuda() for k,v in torch.load(inputs_path,map_location='cpu',weights_only=True).items()}
    repos={'lightforcing':'LightForcing','forcingkv':'Forcing-KV','dasa':'DASA'}
    repo=ROOT/'external_baselines'/repos.get(a.method,'LightForcing')
    if a.method=='dense':
        code=ROOT;config=OmegaConf.load(ROOT/'configs/self_forcing_dmd.yaml')
        default=ROOT/'configs/default_config.yaml'
    elif a.method=='dasa':
        sys.path.insert(0,str(repo/'DASA'))
        code=repo/'DASA/frameworks/self_forcing'
        config=OmegaConf.load(repo/'DASA/configs/self_forcing/inference.yaml')
        default=repo/'DASA/configs/self_forcing/default.yaml'
    else:
        code=repo
        rel='configs/light_forcing_long.yaml' if a.method=='lightforcing' else 'configs/forcing-kv/forcingkv_self_forcing_inference.yaml'
        config=OmegaConf.load(code/rel)
        default=code/'configs/default_config.yaml'
    if default.exists():config=OmegaConf.merge(OmegaConf.load(default),config)
    config.independent_first_frame=False;config.context_noise=0;config.num_frame_per_block=3
    config.height=512;config.width=768;config.num_frames=min(480,a.chunks*12-3)
    config.output_folder=str(out);config.num_output_frames=a.chunks*3
    config.model_kwargs.local_attn_size=15 if a.method=='dasa' else 18
    config.model_kwargs.timestep_shift=5.
    if a.method=='dense':config.model_kwargs.sink_size=0
    if a.method=='forcingkv':
        config.model_kwargs.model_name=str(ROOT/'wan_models/Wan2.1-T2V-1.3B')
        config.forcingkv.offline_head_file=str(repo/'configs_head/forcingkv_heads_selfforcing.json')
    if a.method=='lightforcing':
        config.model_kwargs.efficient_deployment={}
        config.model_kwargs.sparse_config.frame_seq=1536
        # Keep official long-video sink, relative RoPE, HSA and sparsity settings.
    sys.path.insert(0,str(code));os.chdir(code if a.method!='dasa' else repo/'DASA')
    from utils.wan_wrapper import WanDiffusionWrapper
    if a.method=='lightforcing':
        import wan.modules.sparse_attention as sparse
        # Explicit official Triton path on SM90; do not take FA4-missing dense fallback.
        # This is a backend choice, not a changed sparse mask or policy.
        sparse.DEVICE_SM=80
        sc=config.model_kwargs.sparse_config
        sc.sparsity_list=sparse.calculate_chunk_sparsities(a.chunks*3,3,18,OmegaConf.to_container(sc))
    kwargs=OmegaConf.to_container(config.model_kwargs)
    if a.method=='forcingkv':kwargs['method']='forcingkv'
    print(json.dumps({'phase':'loading','method':a.method,'config':OmegaConf.to_container(config)}),flush=True)
    generator=WanDiffusionWrapper(**kwargs,is_causal=True)
    checkpoint=ROOT/'checkpoints/self_forcing_dmd.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    generator.load_state_dict(state['generator_ema'],strict=True);del state
    generator=generator.cuda().to(torch.bfloat16).eval().requires_grad_(False)
    generator.model.gradient_checkpointing=False
    # The existing environment has no FA2. Use the same exact dense-attention
    # mathematical operation in every pipeline; sparse attention is untouched.
    # This affects kernel timing and is explicitly recorded in the result.
    def common_dense(q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,
                     q_scale=None,causal=False,window_size=(-1,-1),**kwargs):
        assert q_lens is None and k_lens is None and window_size==(-1,-1)
        if q_scale is not None:q=q*q_scale
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
            dropout_p=dropout_p,is_causal=causal,scale=softmax_scale).transpose(1,2)
    for name,module in list(sys.modules.items()):
        if name.startswith('wan.modules.'):
            for attr in ('attention','flash_attention'):
                if callable(getattr(module,attr,None)):setattr(module,attr,common_dense)
    for b in generator.model.blocks:
        b.self_attn.max_attention_size=config.model_kwargs.local_attn_size*1536
    # Import the exact upstream pipeline module without importing unrelated
    # training/demo pipelines from its package __init__.
    package=types.ModuleType('pipeline');package.__path__=[str(code/'pipeline')];sys.modules['pipeline']=package
    file='causal_inference_forcingkv.py' if a.method=='forcingkv' else 'causal_inference.py'
    mod=module_file('pipeline.benchmark_official',code/'pipeline'/file)
    cls=getattr(mod,'CausalInferencePipeline_ForcingKV' if a.method=='forcingkv' else 'CausalInferencePipeline')
    pipeline=cls(config,device='cuda',generator=generator,
                 text_encoder=SavedText(inputs['prompt_embeds']),vae=DeferredVAE())
    pipeline.frame_seq_length=1536
    pipeline.eval()
    # Keep the upstream pipeline's random-noise calls, supplying the exact saved
    # transition noise to each call, independent of any operator's RNG use.
    original_randn=torch.randn_like;noise_index=0
    def saved_randn(x,*args,**kwargs):
        nonlocal noise_index
        assert tuple(x.shape)==(3,16,64,96),x.shape
        value=inputs['transition_noise'][noise_index//3,noise_index%3].flatten(0,1)
        noise_index+=1
        return value
    torch.randn_like=saved_randn
    rows=[];calls=[];current={};chunk_started={};layer_number=[0]
    cap=18*1536*12*128*2*2*30
    def storage():
        kv=[];extra=[]
        if pipeline.kv_cache1 is not None:
            for c in pipeline.kv_cache1:
                kv.extend(v for key,v in c.items() if key in ('k','v') or key.endswith(('_k','_v')))
        for block in generator.model.blocks:
            att=block.self_attn
            kv.extend(getattr(att,key,None) for key in ('last_chunk_clean_k','last_chunk_clean_v'))
            extra.extend(getattr(att,key,None) for key in ('last_reference_frame','last_update_step','patch_dynamic_score'))
        if hasattr(generator.model,'x_cache'):extra.extend(generator.model.x_cache.values())
        return dict(self_kv_storage_bytes=storage_bytes(kv),auxiliary_bytes=storage_bytes(extra))
    def before(module,args,kwargs):
        ts=kwargs['timestep'] if 'timestep' in kwargs else args[2]
        start=kwargs['current_start']//1536
        current.update(chunk=start//3,timestep=float(ts[0,0]),layer=0)
        layer_number[0]=0
        if start//3 not in chunk_started:
            torch.cuda.synchronize();chunk_started[start//3]=time.perf_counter()
    def after(module,args,kwargs,result):
        st=storage()
        # DASA reserves 15 main-cache frames + up to 3 cached clean-KV frames.
        assert st['self_kv_storage_bytes']<=cap,(a.method,current,st,cap)
        rows.append(dict(current,**st))
        if current['timestep']==0:
            torch.cuda.synchronize()
            print(json.dumps({'phase':'chunk_complete','method':a.method,'chunk':current['chunk']+1,
                  'seconds':time.perf_counter()-chunk_started[current['chunk']],**st}),flush=True)
            save(out/f'{a.method}.cache_trace.json',rows)
    h1=generator.register_forward_pre_hook(before,with_kwargs=True)
    h2=generator.register_forward_hook(after,with_kwargs=True)
    layer_hooks=[]
    for index,block in enumerate(generator.model.blocks):
        def layer_hook(module,args,_index=index):current['layer']=_index
        layer_hooks.append(block.self_attn.register_forward_pre_hook(layer_hook))
    causal=sys.modules[generator.model.__class__.__module__]
    original_attention=causal.attention
    research=None
    if a.research_config:
        assert a.method=='dense', 'Research hooks use only the native dense pipeline'
        from local_experiments.cache_research import ResearchAttention
        research=ResearchAttention(json.loads(Path(a.research_config).read_text()),out,current,original_attention)
    def observed(q,k,v,*args,**kwargs):
        calls.append(dict(current,q_tokens=q.shape[1],k_tokens=k.shape[1],heads=q.shape[2],sparse=False))
        assert k.shape[1]<=18*1536,(a.method,k.shape)
        return (research or original_attention)(q,k,v,*args,**kwargs)
    causal.attention=observed
    if a.method=='lightforcing':
        original_sparse=causal.sparse_attention
        def observed_sparse(q,k,v,*args,**kwargs):
            calls.append(dict(current,q_tokens=q.shape[1],k_tokens=k.shape[1],heads=q.shape[2],sparse=True,
                         requested_sparsity=float(kwargs['sparsity_list'][kwargs['chunk_id']])))
            assert k.shape[1]<=18*1536
            return original_sparse(q,k,v,*args,**kwargs)
        causal.sparse_attention=observed_sparse
    from local_experiments.stream_eval_common import benchmark
    def reset_hidden():
        for b in generator.model.blocks:
            att=b.self_attn
            for attr in ['last_reference_frame','last_update_step','last_chunk_clean_k','last_chunk_clean_v','patch_dynamic_score']:
                if hasattr(att,attr):setattr(att,attr,None)
            if hasattr(att,'current_frame_num'):att.current_frame_num=0
        if hasattr(generator.model,'x_cache'):generator.model.x_cache.clear()
    def generate():
        nonlocal noise_index
        noise_index=0;reset_hidden()
        _,latent=pipeline.inference(inputs['initial_noise'][:,:a.chunks*3],['saved prompt'],return_latents=True,profile=False)
        assert noise_index==a.chunks*3
        return latent
    def after_warm():
        h1.remove();h2.remove()
        for handle in layer_hooks:handle.remove()
        causal.attention=original_attention
        if a.method=='lightforcing':causal.sparse_attention=original_sparse
        save(out/'warm_audit.json',dict(max_self_kv_storage_bytes=max(r['self_kv_storage_bytes'] for r in rows),max_auxiliary_bytes=max(r['auxiliary_bytes'] for r in rows),calls=len(calls),max_attended_tokens=max(r['k_tokens'] for r in calls),history_cap_frames=15,current_frames=3))
    metadata=dict(method=a.method,checkpoint_sha256=digest(checkpoint),inputs_sha256=digest(inputs_path),height=512,width=768,chunks=41,frames=480,fps=16,history_chunk_cap=5,config=OmegaConf.to_container(config),pipeline_sha256=digest(code/'pipeline'/file),causal_sha256=digest(causal.__file__),benchmark_script_sha256=digest(__file__),scope='Original complete upstream pipeline, common native VAE; native cache/position policy retained; no added rebase. Same base checkpoint; method-specific fine-tuning not reproduced.')
    try:benchmark(a.method,out,generate,metadata,after_warm,storage)
    finally:torch.randn_like=original_randn

if __name__=='__main__':main()
