"""Use the release benchmark protocol; only base selects the experimental CUDA kernel."""
import ctypes,json,sys
from pathlib import Path
from local_experiments.video_runtime_v1 import deployment
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.benchmark_hotspots import run
tier=sys.argv[1]
original=deployment.make_runtime
def make_runtime(*args):
    rt=original(*args)
    if tier=='base':
        fn=ctypes.CDLL('/tmp/jzk_tailmask_lane_fma.so').run
        fn.argtypes=rt.packed_attention.kernel.argtypes
        rt.packed_attention.kernel=fn
    return rt
deployment.make_runtime=make_runtime
out=Path('/tmp/jzk_lanefma_105_'+tier)
cfg=config(tier,local_assets='/tmp/streaming_sparse-0/runtime',output=out,reps=5,chunks=9,skip_profile=True,block_graph=False,capture_route=False)
run(cfg)
r=json.loads((out/'result.json').read_text())
stats=r['runtime_stats']
result=dict(tier=tier,kernel='lane_fma' if tier=='base' else 'deployed',seconds=r['seconds'],median_s=r['median_seconds'],cuda_calls=stats['packed_attention_calls'],direct_commits=stats['direct_layout_commits'],counters=stats['counters'],video=r['video'],frames=r['frames'])
(Path(__file__).parent/f'tier_{tier}.json').write_text(json.dumps(result,indent=2))
