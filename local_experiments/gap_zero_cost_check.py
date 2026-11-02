"""Invalid-output counterfactual timing: remove eligible dense attention only.
This is NOT a sparse implementation or a quality candidate. It estimates the
remaining backbone/VAE workload and includes allocation of zero outputs.
"""
import torch,json,time
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk,timed_run
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard,RunLock,digest
@torch.no_grad()
def main():
    root=Path('artifacts/gap_zero_cost_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        from utils.wan_wrapper import WanVAEWrapper
        import wan.modules.causal_model as causal
        original=causal.attention
        identity=dict(script=digest(Path(__file__)),scope='Diagnostic invalid outputs, no video quality interpretation; zero-output replacement still allocates and launches memset. Warmed complete generation+VAE; matched dense controls.')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda();rt=Runtime(engine,'dense');count=[0]
        def zero(q,k,v,*args,**kwargs):
            meta=engine.current
            if q.shape[1]==4608 and k.shape[1]>=4608 and meta['layer']>0 and meta['chunk']>=2 and meta['step']<4:
                count[0]+=1;return torch.zeros_like(q)
            return original(q,k,v,*args,**kwargs)
        for name in ['dense','zero']:
            causal.attention=original if name=='dense' else zero
            rt.reset()
            for c in range(6):z=generate_chunk(engine,emb,101,c)
            vae.model.clear_cache();vae.decode_to_pixel(z,use_cache=True);vae.decode_to_pixel(z,use_cache=True)
            timed_run(engine,rt,vae,emb,101)
        for unit in ['dense_0','zero_0','zero_1','dense_1']:
            if (root/(unit+'.json')).exists():continue
            assert not device_peers();disk_guard(root,50,1);causal.attention=zero if unit.startswith('zero') else original;count[0]=0
            atomic(root/'status.json',dict(status='running',unit=unit));zs,frames,perf=timed_run(engine,rt,vae,emb,101)
            if unit.startswith('zero'):assert count[0]==4524,count
            assert not device_peers();atomic(root/(unit+'.json'),dict(unit=unit,replaced_calls=count[0],performance=perf,valid_generation=False if unit.startswith('zero') else True));print(json.dumps(dict(unit=unit,seconds=perf['total_generation_vae_seconds'],replaced=count[0])),flush=True);del zs,frames
        atomic(root/'status.json',dict(status='complete'))
    except Exception as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if 'original' in locals():causal.attention=original
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
