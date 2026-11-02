"""Alternating already-warmed runtimes; no peer GPU processes allowed."""
import gc,json,os,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_gain_vector64_runtime import GapGainVector64Runtime as GapFeatureRuntime
from .gap_reorder_runtime import GapReorderRuntime as GapSummaryReuseRuntime
from .gap_gain_runtime import GapGainRuntime as GapFlatOrderedRuntime
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk,timed_run
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .gap_group_isolated import device_peers
from .gap_peer_watch import PeerWatch
ROOT=Path('artifacts/gap_reorder_20261102/performance_gpu7')
MODEL=Path('artifacts/four_gaps_20261031/shared/epoch1.pt')
@torch.no_grad()
def main():
    check=json.loads(Path('artifacts/gap_reorder_20261102/checks.json').read_text())
    assert check['status']=='complete' and check.get('peer_monitor_validated'), 'Clean local validation required first'
    assert all(c['exact'] for r in check['rows'] for c in r['candidates']), 'Gather must be bit preserving'
    lock=RunLock(ROOT);runtimes={};bindings={}
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert os.environ.get("GAP_TAIL_MODE")=="1"
        identity=dict(tail_mode=int(os.environ.get('GAP_TAIL_MODE','3')),model=digest(MODEL),gpu=os.environ['CUDA_VISIBLE_DEVICES'],case='horse_field',seed=101,code={n:digest(Path('local_experiments')/n) for n in ['gap_reorder_perf.py','gap_reorder_runtime.py','gap_reorder_kernel.py','gap_gain_vector64_runtime.py','gap_gain_project_vector.py','gap_gain_project_ops.py','gap_gain_vector64_perf.py','gap_peer_watch.py','gap_route_qp_perf.py','gap_route_qp_runtime.py','gap_route_query_project.py','gap_staged_history_runtime.py','gap_staged_single_runtime.py','gap_gpu_processes.py','gap_group_isolated.py','gap_group_fused_features.py','gap_group_projection.py','gap_route_staged_graph_perf.py','gap_route_staged_graph_runtime.py','gap_route_staging.py','gap_tail_single_perf.py','gap_tail_single_runtime.py','gap_tail_single.py','gap_pool_linear_perf.py','gap_pool_linear_runtime.py','gap_pool_linear.py','gap_route_half_perf.py','gap_route_half_runtime.py','gap_route_fused_tuned_perf.py','gap_route_fused_tuned_runtime.py','gap_route_fused_stats_perf.py','gap_route_fused_stats_runtime.py','gap_route_fused_stats_ops.py','gap_route_bf16_stats.py','gap_weight_linear.py','gap_tail_packed_perf.py','gap_tail_packed_runtime.py','gap_tail_packed.py','gap_route_packed.py','gap_exact_round_perf.py','gap_exact_round_runtime.py','gap_exact_round_kernel.py','gap_history_cov_perf.py','gap_history_cov_runtime.py','gap_covariance_kernel.py','gap_route_combined_runtime.py','gap_route_pick_ops.py','gap_route_pick.py','gap_route_bf16qk.py','gap_tail_bf16qk_stage2_runtime.py','gap_tail_bf16qk_stage2.py','gap_route_reduce_runtime.py','gap_route_reduce.py','gap_route_contiguous_ops.py','gap_route_score_scalar.py','gap_tensor_gram_runtime.py','gap_tensor_gram.py','gap_group_fused_runtime.py','gap_group_fused.py','gap_feature_pool_runtime.py','gap_feature_pool_ops.py','gap_feature_pool_kernel.py','gap_summary_lite_runtime.py','gap_covariance_kernel.py','gap_raw_pool_ops.py','gap_pool_weights_kernel.py', 'gap_route_tensor_runtime.py', 'gap_route_tensorcore.py', 'gap_precision_runtime.py', 'gap_precision_tail.py', 'gap_direct_summary_runtime.py', 'gap_direct_summary_ops.py', 'gap_direct_center_kernel.py']})
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        engine=Engine();blocks=engine.wrapper.model.blocks
        from utils.wan_wrapper import WanVAEWrapper
        vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        ref=torch.load('artifacts/gap_gain_vector64_20261102/performance_gpu7/after.latents.pt',weights_only=True)
        def measure(unit,rt):
            dest=ROOT/(unit+'.json')
            if dest.exists():return
            disk_guard(ROOT,50,1);peers=device_peers();assert not peers,peers
            atomic(ROOT/'status.json',dict(status='running',unit=unit))
            with PeerWatch() as watch:
                zs,frames,perf=timed_run(engine,rt,vae,emb,101)
            atomic(ROOT/(unit+'.peer_samples.json'),watch.summary())
            watch.validate()
            peers=device_peers();assert not peers,peers
            exact=None if unit.startswith('dense') else torch.equal(torch.cat(zs,1).cpu(),ref)
            if unit.startswith(('before','after','ordered')):assert exact,'Exact vector route full rollout changed'
            latent_nmse=None if unit.startswith('dense') else float((torch.cat(zs,1).cpu().float()-ref.float()).square().sum()/ref.float().square().sum())
            flat_exact=None
            if unit=='after_0':atomic(ROOT/'after.latents.pt',torch.cat(zs,1).cpu(),True)
            if unit.startswith('ordered'):
                flat_exact=torch.equal(torch.cat(zs,1).cpu(),ref);assert flat_exact,'Exact pooled reduction changed full video'
            atomic(dest,dict(unit=unit,performance=perf,flat_branch_latents_exact=flat_exact,full_latents_exact=exact,latent_nmse_from_accepted=latent_nmse,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],no_peers_at_boundaries=True,note='Peak includes inactive warmed runtime workspaces; not deployment peak.'))
            print(json.dumps(dict(unit=unit,seconds=perf['total_generation_vae_seconds'],exact=exact)),flush=True)
        def warm(rt):
            for c in range(6):z=generate_chunk(engine,emb,101,c)
            vae.model.clear_cache();vae.decode_to_pixel(z,use_cache=True);vae.decode_to_pixel(z,use_cache=True)
        rt=Runtime(engine,'dense');warm(rt);timed_run(engine,rt,vae,emb,101);measure('dense_0',rt);rt.close();del rt
        for name,cls in [('before',GapFeatureRuntime),('after',GapSummaryReuseRuntime)]:
            atomic(ROOT/'status.json',dict(status='warming',unit=name))
            rt=cls(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
            rt.model.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=False)['model']);warm(rt)
            # One full unmeasured rollout warms all VAE/runtime shapes, not only six chunks.
            timed_run(engine,rt,vae,emb,101)
            bindings[name]=[(b.forward,b.self_attn.forward,b.self_attn.max_attention_size) for b in blocks]
            runtimes[name]=rt;rt.close()
        # Both sparse implementations share exactly the same raw KV buffers.
        for unit in ['before_0','after_0','after_1','before_1']:
            name=unit.split('_')[0];rt=runtimes[name]
            for b,(bf,af,size) in zip(blocks,bindings[name]):b.forward=bf;b.self_attn.forward=af;b.self_attn.max_attention_size=size
            measure(unit,rt)
        for rt in runtimes.values():rt.close()
        runtimes.clear();bindings.clear();del rt;gc.collect();torch.cuda.empty_cache()
        rt=Runtime(engine,'dense');warm(rt);timed_run(engine,rt,vae,emb,101);measure('dense_1',rt);rt.close();del rt
        atomic(ROOT/'status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        for rt in runtimes.values():rt.close()
        lock.close()
if __name__=='__main__':main()
