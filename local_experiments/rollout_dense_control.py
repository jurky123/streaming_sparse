"""Exact native chronological dense cache with host indices; separate frozen control."""
import argparse,gc,json,time,types
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_runtime import Runtime,ROOT
from .rollout_suite import generate_chunk,timed_run,guard
from .learned_cache.io import atomic,RunLock,digest

class HostDense(Runtime):
    def __init__(self,engine):
        super().__init__(engine,'dense');self.method='dense_host';self.ends=[0]*30;self.locals=[0]*30
        for l,b in enumerate(engine.wrapper.model.blocks):
            if l==0:continue # Match sparse runtime's unchanged layer0.
            def forward(module,x,seq_lens,grid_sizes,freqs,block_mask,kv_cache=None,current_start=0,cache_start=None,index=l):
                return self.host_forward(index,module,x,grid_sizes,freqs,kv_cache,current_start)
            b.self_attn.forward=types.MethodType(forward,b.self_attn)
    def reset(self):
        super().reset();self.ends=[0]*30;self.locals=[0]*30
    @torch.no_grad()
    def host_forward(self,l,module,x,grid_sizes,freqs,cache,start):
        from wan.modules.causal_model import causal_rope_apply
        b,s,_=x.shape;assert b==1 and s==4608 and module.sink_size==0
        q=module.norm_q(module.q(x)).view(b,s,12,128)
        k=module.norm_k(module.k(x)).view(b,s,12,128);v=module.v(x).view(b,s,12,128)
        q=causal_rope_apply(q,grid_sizes,freqs,start_frame=start//1536).type_as(v)
        k=causal_rope_apply(k,grid_sizes,freqs,start_frame=start//1536).type_as(v)
        end=start+s;size=cache['k'].shape[1];old=self.locals[l]
        if end>self.ends[l] and s+old>size:
            evict=s+old-size;rolled=old-evict
            for key in ['k','v']:cache[key][:,:rolled]=cache[key][:,evict:evict+rolled].clone()
            local=old+end-self.ends[l]-evict
        else:local=old+end-self.ends[l]
        cache['k'][:,local-s:local]=k;cache['v'][:,local-s:local]=v
        out=self.engine.dense(q,cache['k'][:,max(0,local-module.max_attention_size):local],cache['v'][:,max(0,local-module.max_attention_size):local])
        self.ends[l]=end;self.locals[l]=local
        cache['global_end_index'].fill_(end);cache['local_end_index'].fill_(local)
        return module.o(out.flatten(2))

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    root=ROOT/'dense_control';lock=RunLock(root)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        emb=torch.load('artifacts/mass_variance_20261030/embeddings/basketball_court.pt',weights_only=True).cuda()
        identity=dict(code={f:digest(Path('local_experiments')/f) for f in ['rollout_dense_control.py','rollout_runtime.py','rollout_kernels.py','rollout_suite.py','broad_common.py']},repeats=a.repeats,case='basketball_court',seed=606,gpu=torch.cuda.get_device_name(),note='Separate paired control; layer0 remains native. All other dense cache tokens, ordering, copies, RoPE and attention unchanged; host indices remove GPU scalar reads.')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        methods=['dense','dense_host','mass_var']
        for rep in range(a.repeats):
            for method in methods[rep:]+methods[:rep]:
                path=root/f'{method}.rep{rep}.json'
                if path.exists():continue
                guard();gc.collect();torch.cuda.empty_cache()
                rt=HostDense(engine) if method=='dense_host' else Runtime(engine,method)
                try:
                    for ch in range(6):warm=generate_chunk(engine,emb,606,ch)
                    vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                    latents,frames,result=timed_run(engine,rt,vae,emb,606);value=torch.cat(latents,1).cpu()
                    if method=='dense_host':
                        target=torch.load(root/'dense.latents.pt',weights_only=True)
                        assert torch.equal(value,target),'host-index baseline changed output'
                        result['bit_exact_native_dense']=True
                    elif rep==0:atomic(root/f'{method}.latents.pt',value,True)
                    else:assert torch.equal(value,torch.load(root/f'{method}.latents.pt',weights_only=True))
                    result.update(rep=rep,completed=True);atomic(path,result)
                    print(json.dumps(dict(method=method,rep=rep,total_s=result['total_generation_vae_seconds'],generation_s=result['generation_seconds'],exact=result.get('bit_exact_native_dense'))),flush=True)
                    del value,latents,frames
                finally:rt.close()
        atomic(root/'status.json',dict(completed=True))
    except Exception as exc:
        atomic(root/'status.json',dict(completed=False,error=repr(exc)));raise
    finally:lock.close()

if __name__=='__main__':main()
