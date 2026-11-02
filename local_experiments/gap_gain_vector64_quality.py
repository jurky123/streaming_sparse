"""Fused history features: early-video quality against measured SingleTail."""
import gc,json,os,signal,torch
from pathlib import Path
from .broad_common import Engine,cases,ROOT as DATA
from .gap_gain_vector64_runtime import GapGainVector64Runtime as GapFlatRuntime
VARIANT="gain_vector64"
from .rollout_suite import generate_chunk,video
from .high_sparse_experiment import prefix_decode,prefix_quality
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/gap_gain_vector64_20261102/quality');OLD=Path('artifacts/four_gaps_20261031');MODEL=OLD/'shared/epoch1.pt'
@torch.no_grad()
def main():
    check=json.loads(Path('artifacts/gap_gain_vector64_20261102/checks.json').read_text())
    assert check['status']=='complete' and check.get('peer_monitor_validated'), 'Clean local validation required first'
    assert all(r['contribution_nmse'] < 1.e-8 and r['error_nmse'] < 1.e-8 for r in check['rows']), 'Investigate local numerical error first'
    lock=RunLock(ROOT);stop=[False];rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        samples=[c|dict(seed=seed,id=f'test_{c["name"]}_s{seed}') for seed in [404,505] for c in cases('test') if c['seed']==202]
        samples=[c for c in samples if (c['name'],c['seed']) in [('bird_flight',505),('traffic_crossing',404)]]
        assert os.environ.get("GAP_TAIL_MODE")=="1"
        ident=dict(variant=VARIANT,tail_mode=1,model=digest(MODEL),samples=samples,window=[21,101],screen=dict(mean_lpips_delta_max=.005,mean_ssim_delta_min=-.005,case_lpips_delta_max=.02,case_ssim_delta_min=-.02),scope='Two-case early RGB comparison against RouteQp and matched dense. Legacy screen is diagnostic only; user permits quality/speed tradeoffs. Recovery training has not been performed.',code={n:digest(Path('local_experiments')/n) for n in ['gap_gain_vector64_quality.py','gap_gain_vector64_runtime.py','gap_gain_project_vector.py','gap_gain_project_ops.py','gap_route_qp_quality.py','gap_route_qp_runtime.py','gap_route_query_project.py','gap_staged_history_runtime.py','gap_staged_single_runtime.py','gap_gpu_processes.py','gap_group_isolated.py','gap_group_fused_features.py','gap_group_projection.py','gap_route_staged_graph_quality.py','gap_route_staged_graph_runtime.py','gap_route_staging.py','gap_tail_single_quality.py','gap_tail_single_runtime.py','gap_tail_single.py','gap_pool_linear_quality.py','gap_pool_linear_runtime.py','gap_pool_linear.py','gap_route_half_quality.py','gap_route_half_runtime.py','gap_route_fused_tuned_quality.py','gap_route_fused_tuned_runtime.py','gap_route_fused_stats_quality.py','gap_route_fused_stats_runtime.py','gap_route_fused_stats_ops.py','gap_route_bf16_stats.py','gap_weight_linear.py','gap_tail_packed_quality.py','gap_tail_packed_runtime.py','gap_tail_packed.py','gap_route_packed.py','gap_exact_round_quality.py','gap_exact_round_runtime.py','gap_exact_round_kernel.py','gap_history_cov_quality.py','gap_history_cov_runtime.py','gap_covariance_kernel.py','gap_route_combined_runtime.py','gap_route_bf16qk.py','gap_route_pick_runtime.py','gap_route_pick_ops.py','gap_route_pick.py','gap_tail_bf16qk_stage2_runtime.py','gap_tail_bf16qk_stage2.py','gap_tail_bf16qk_runtime.py','gap_tail_bf16qk.py','gap_route_reduce_runtime.py','gap_route_reduce.py','gap_route_contiguous_ops.py','gap_route_score_runtime.py','gap_route_score_scalar_ops.py','gap_route_score_scalar.py','gap_tensor_gram_runtime.py','gap_tensor_gram.py','gap_group_fused_runtime.py','gap_group_fused.py','gap_feature_pool_runtime.py','gap_feature_pool_ops.py','gap_feature_pool_kernel.py','gap_summary_lite_runtime.py','gap_covariance_kernel.py','gap_raw_pool_ops.py','gap_pool_weights_kernel.py','gap_route_tensor_runtime.py','gap_route_tensorcore.py','gap_precision_runtime.py','gap_precision_tail.py','gap_direct_summary_runtime.py','gap_direct_summary_ops.py','gap_direct_center_kernel.py']})
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==ident
        else:atomic(ROOT/'identity.json',ident)
        os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
        import lpips
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        rt=GapFlatRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=False)['model'])
        for case in samples:
            folder=ROOT/case['id'];folder.mkdir(exist_ok=True);dest=folder/'result.json'
            if dest.exists():continue
            if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted'));return
            disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache()
            old=OLD/f'test_s{case["seed"]}'/case['id']
            def decode(path):
                z=torch.load(path,weights_only=True).cuda();return prefix_decode(vae,list(z.split(3,1)))
            dense=decode(old/'dense.latents.pt');base=decode(Path('artifacts/gap_route_qp_20261102/quality')/case['id']/'flat.latents.pt')
            baseline=prefix_quality(base,dense,metric)
            emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
            rt.reset()
            try:
                atomic(ROOT/'status.json',dict(status='running',case=case['id']))
                zs=[generate_chunk(engine,emb,case['seed'],c) for c in range(9)];frames=prefix_decode(vae,zs)
                quality=prefix_quality(frames,dense,metric);drift=prefix_quality(frames,base,metric)
                atomic(folder/'flat.latents.pt',torch.cat(zs,1).cpu(),True);video(folder/'flat.mp4',frames)
                delta={k:quality[k]-baseline[k] for k in ['lpips','ssim','psnr']}
                atomic(dest,dict(case=case,baseline=baseline,flat=quality,delta=delta,drift_from_native_route=drift,runtime=rt.finish(),baseline_video=str(Path('artifacts/gap_route_qp_20261102/quality')/case['id']/'flat.mp4')))
                print(json.dumps(dict(case=case['id'],delta=delta,drift_lpips=drift['lpips'])),flush=True)
                del zs,frames,dense,base
            finally:pass
        rows=[json.loads((ROOT/c['id']/'result.json').read_text()) for c in samples]
        mean={k:sum(r['delta'][k] for r in rows)/len(rows) for k in ['lpips','ssim','psnr']}
        passed=mean['lpips']<=.005 and mean['ssim']>=-.005 and all(r['delta']['lpips']<=.02 and r['delta']['ssim']>=-.02 for r in rows)
        atomic(ROOT/'summary.json',dict(cases=len(rows),mean_delta=mean,passed_screen=passed,acceptance_policy='diagnostic only; evaluate measured speed and quality jointly',rows=[dict(case=r['case']['id'],delta=r['delta'],drift_lpips=r['drift_from_native_route']['lpips']) for r in rows],scope=ident['scope']))
        atomic(ROOT/'status.json',dict(status='complete',passed_screen=passed))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
