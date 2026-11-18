"""Each warp loads one 16-page mask per lane and broadcasts the current tile."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
s=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
old='    auto compute_tile=[&](int step, auto shape){'
assert s.count(old)==1
s=s.replace(old,'''    unsigned owned=mask[l/2]>>(16*(l%2));
'''+old).replace('unsigned lane_mask=mask[pg/32]>>((pg%32)+2*z);','unsigned lane_mask=__shfl_sync(~0u,owned,step)>>(2*z);')
out=Path('/tmp/jzk_page_maskreg.cu');out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
