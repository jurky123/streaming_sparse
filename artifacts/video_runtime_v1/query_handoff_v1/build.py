"""Pass the exact phase's existing SM80 A registers into the tail QK."""
from pathlib import Path
import subprocess
source=Path(__file__).resolve().parents[3]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu'
s=source.read_text()
a=s.index('void tail('); b=s.index('    // One slab',a)
part=s[a:b].replace('const bf* Q,','const ::A (&q)[8],').replace('    A q[8];C am','    C am')
s=s[:a]+part[:part.index('    #pragma unroll')]+s[b:]
key='                mma(score[n],q[k],B{{v.x,v.y}});mma(score[n+1],q[k],B{{v.z,v.w}});'
assert s.count(key)==1
s=s.replace(key,'''                A query={{q[k].x[0],q[k].x[1],q[k].x[2],q[k].x[3]}};
                mma(score[n],query,B{{v.x,v.y}});mma(score[n+1],query,B{{v.z,v.w}});''').replace('tailops::tail(Q,','tailops::tail(q,')
out=Path('/tmp/jzk_page_qhandoff.cu');out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
