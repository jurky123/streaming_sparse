import importlib,sys,types
from pathlib import Path
import torch
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2]
EXT=ROOT/'external_baselines'

def install_method(method,engine):
    import wan.modules.causal_model as causal
    orig=causal.attention
    counts=[0]
    if method=='lightforcing':
        pkg=types.ModuleType('anchor_light_kernels')
        pkg.__path__=[str(EXT/'LightForcing/wan/modules')]
        sys.modules[pkg.__name__]=pkg
        light=importlib.import_module(pkg.__name__+'.sparse_attention')
        assert light.DEVICE_SM==80
    if method=='pisa':
        sys.path.insert(0,str(EXT/'PISA'))
        from piecewise_attn import piecewise_sparse_attention
    if method=='solattn':
        sys.path.insert(0,str(EXT/'Sana/techniques/sparse_backends'))
        from sol_attn.triton_ref import sol_attn
    if method=='svgear':
        from ear import EAR
        ear=EAR()
    def prepare(n):
        nonlocal kwargs
        counts[0]=0
        if method=='svgear':ear.reset()
        if method=='lightforcing':
            kind='short' if n==7 else 'long'
            conf=OmegaConf.load(EXT/f'LightForcing/configs/light_forcing_{kind}.yaml').model_kwargs.sparse_config
            kwargs=dict(sparsity_list=light.calculate_chunk_sparsities(3*n,3,18,conf),frame_seq=1536,**{k:conf[k] for k in ('keep_sink','keep_near','keep_frames')})
    kwargs={}
    def dispatch(q,k,v,*args,**kw):
        c,l,s=(engine.current[x] for x in ('chunk','layer','step'))
        if method=='lightforcing':
            counts[0]+=1
            return light.sparse_attention(q,k,v,chunk_id=c,**kwargs)
        if c<2 or s==4 or l==0:return orig(q,k,v,*args,**kw)
        counts[0]+=1
        if method=='svgear':return ear(q,k,v,l)
        padded=torch.zeros_like(k);padded[:,-q.shape[1]:]=q
        if method=='pisa':
            out=piecewise_sparse_attention(padded.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),density=.1).transpose(1,2)
        else:
            out=sol_attn(padded.contiguous(),k.contiguous(),v.contiguous(),tau=1.,sink_start=k.shape[1]-q.shape[1],sink_tokens=q.shape[1])
        return out[:,-q.shape[1]:].contiguous()
    causal.attention=dispatch
    return prepare,counts
