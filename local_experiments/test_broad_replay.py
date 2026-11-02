import json
import sys
import types
from pathlib import Path
import torch
from omegaconf import OmegaConf
from .broad_common import ROOT,cases,Engine,noise
from .learned_cache.io import atomic
from .run_official_baseline import module_file,SavedText,DeferredVAE

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=cases('train')[0];folder=ROOT/'trajectories'/case['id'];engine=Engine()
    saved=torch.load(folder/'chunk05.pt',weights_only=True);expected=saved['queries'];checks=[]
    def callback(c,q,k,v):
        ids=torch.linspace(0,q.shape[1]-1,16,device='cuda').long();actual=q[0,ids].transpose(0,1).cpu()
        ref=expected[c['step'],c['layer']]
        checks.append(dict(step=c['step'],layer=c['layer'],equal=torch.equal(actual,ref),max_abs=(actual.float()-ref.float()).abs().max().item()))
    engine.replay(case,callback,stop_chunk=5)
    assert len(checks)==120 and all(r['equal'] for r in checks),max(r['max_abs'] for r in checks)
    emb=torch.load(ROOT/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
    code=Path.cwd();package=types.ModuleType('pipeline');package.__path__=[str(code/'pipeline')];sys.modules['pipeline']=package
    module=module_file('pipeline.broad_native_check',code/'pipeline/causal_inference.py')
    config=OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'),OmegaConf.load('configs/self_forcing_dmd.yaml'))
    config.independent_first_frame=False;config.context_noise=0;config.num_frame_per_block=3;config.model_kwargs.local_attn_size=18
    pipe=module.CausalInferencePipeline(config,device='cuda',generator=engine.wrapper,text_encoder=SavedText(emb),vae=DeferredVAE());pipe.frame_seq_length=1536
    initial=torch.cat([noise(case['seed'],chunk,0) for chunk in range(2)],1);index=[0];original=torch.randn_like
    def injected(x,*args,**kwargs):
        i=index[0];index[0]+=1;return noise(case['seed'],i//3,i%3+1).flatten(0,1)
    torch.randn_like=injected
    try:_,native=pipe.inference(initial,['saved'],return_latents=True)
    finally:torch.randn_like=original
    captured=torch.cat([torch.load(folder/f'chunk{c:02d}.pt',weights_only=True)['clean'] for c in range(2)],1).cuda()
    assert index[0]==6 and torch.equal(native,captured),(index[0],(native-captured).abs().max())
    atomic(ROOT/'replay_checks.json',dict(native_two_chunks_bit_exact=True,clean_replay_query_bit_exact=True,
        checked_layers=30,checked_heads=12,denoise_steps=4,queries_per_head=16,records=checks))
    print('Native generation and clean-prefix replay bit-exact checks passed',flush=True)

if __name__=='__main__':main()
