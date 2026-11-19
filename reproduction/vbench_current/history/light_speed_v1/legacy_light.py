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

ROOT=Path('/home/shuchenweng/jzk/streaming_sparse')


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


def build(chunks):
    a=types.SimpleNamespace(method="lightforcing",chunks=chunks)
    out=Path("/mnt/data/vbench_run_20261118")
    inputs={"prompt_embeds":None}
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
    return pipeline
