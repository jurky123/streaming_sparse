"""Alternating already-warmed runtimes; no peer GPU processes allowed."""
import gc,json,os,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_feature_runtime import GapFeatureRuntime
from .gap_summary_reuse_runtime import GapSummaryReuseRuntime
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk,timed_run
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .gap_group_isolated import device_peers
ROOT=Path(os.environ.get('GAP_HOT_PAIR_ROOT','artifacts/gap_hotspots_20261031/hot_pair'))
MODEL=Path('artifacts/four_gaps_20261031/shared/epoch1.pt')
@torch.no_grad()
def main():
    lock=RunLock(ROOT);runtimes={};bindings={}
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity=dict(model=digest(MODEL),gpu=os.environ['CUDA_VISIBLE_DEVICES'],case='horse_field',seed=101,code={n:digest(Path('local_experiments')/n) for n in ['gap_hot_pair.py','gap_feature_runtime.py','gap_summary_reuse_runtime.py','gap_group_runtime.py','gap_group_norm.py','gap_commit_graph_runtime.py']})
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        engine=Engine();blocks=engine.wrapper.model.blocks
        from utils.wan_wrapper import WanVAEWrapper
        vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        ref=torch.load('artifacts/gap_profile_20261031/performance_features/reference.latents.pt',weights_only=True)
        def measure(unit,rt):
            dest=ROOT/(unit+'.json')
            if dest.exists():return
            disk_guard(ROOT,50,1);peers=device_peers();assert not peers,peers
            atomic(ROOT/'status.json',dict(status='running',unit=unit))
            zs,frames,perf=timed_run(engine,rt,vae,emb,101)
            peers=device_peers();assert not peers,peers
            exact=None if unit.startswith('dense') else torch.equal(torch.cat(zs,1).cpu(),ref)
            if not unit.startswith('dense'):assert exact,'Actual full rollout changed'
            atomic(dest,dict(unit=unit,performance=perf,full_latents_exact=exact,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],no_peers_at_boundaries=True,note='Peak includes inactive warmed runtime workspaces; not deployment peak.'))
            print(json.dumps(dict(unit=unit,seconds=perf['total_generation_vae_seconds'],exact=exact)),flush=True)
        def warm(rt):
            for c in range(6):z=generate_chunk(engine,emb,101,c)
            vae.model.clear_cache();vae.decode_to_pixel(z,use_cache=True);vae.decode_to_pixel(z,use_cache=True)
        rt=Runtime(engine,'dense');warm(rt);measure('dense_0',rt);rt.close();del rt
        for name,cls in [('before',GapFeatureRuntime),('after',GapSummaryReuseRuntime)]:
            atomic(ROOT/'status.json',dict(status='warming',unit=name))
            rt=cls(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
            rt.model.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=False)['model']);warm(rt)
            bindings[name]=[(b.forward,b.self_attn.forward,b.self_attn.max_attention_size) for b in blocks]
            runtimes[name]=rt;rt.close()
        # Both sparse implementations share exactly the same raw KV buffers.
        for unit in ['before_0','after_0','after_1','before_1','before_2','after_2']:
            name=unit.split('_')[0];rt=runtimes[name]
            for b,(bf,af,size) in zip(blocks,bindings[name]):b.forward=bf;b.self_attn.forward=af;b.self_attn.max_attention_size=size
            measure(unit,rt)
        for rt in runtimes.values():rt.close()
        runtimes.clear();bindings.clear();del rt;gc.collect();torch.cuda.empty_cache()
        rt=Runtime(engine,'dense');warm(rt);measure('dense_1',rt);rt.close();del rt
        atomic(ROOT/'status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        for rt in runtimes.values():rt.close()
        lock.close()
if __name__=='__main__':main()
