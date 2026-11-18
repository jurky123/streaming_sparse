"""Prefer the 24 common exact pages in L2 on SM80."""
from pathlib import Path
import subprocess
src = Path(__file__).resolve().parents[3]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu'
s = src.read_text()
i = s.index('    for(int step=-1;step<count;step++)')
s = s[:i] + '''    unsigned long long keep,normal;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(keep));
    asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(normal));
''' + s[i:]
s = s.replace('            int slot=(step+1+count)&1;', '''            // list[0] owns the current query page; list[1..24] contains the common frame.
            auto policy=(step>=0 && step<24)?keep:normal;
            int slot=(step+1+count)&1;''')
for x in ('k','v'):
    orig=f'asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(a{x}),"l"(f{x}+m));'
    assert s.count(orig)==1
    s=s.replace(orig,orig.replace('.global ','.global.L2::cache_hint ').replace('16;"','16,%2;"').replace('+m));','+m),"l"(policy));'))
out=Path('/tmp/jzk_anchor_cache.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
