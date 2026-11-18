"""Express both tile shapes with one inline local function in production CUDA."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
src=(root/'control.cu').read_text()
s=src.replace('#include <stdint.h>','#include <stdint.h>\n#include <type_traits>')
a=s.index('    for(int step=1;step<tiles;step++){')
b=s.index('    // The opposite summary slot',a)
body=s[a:b].replace('for(int step=1;step<tiles;step++){','auto compute_tile=[&](int step, auto shape){\n        constexpr bool first=decltype(shape)::value;')
body=body.replace('for(int n=0;n<8;n+=2)','for(int n=first ? 4 : 0;n<8;n+=2)')
body=body.replace('float v=(lane_mask &','float v=(first && n<4) || (lane_mask &')
assert body.endswith('    }\n')
body=body[:-6]+'''    };
    // Pages 16..23 are exact. Their 32 representatives occupy QK fragments 0..3.
    compute_tile(1,std::true_type{});
    for(int step=2;step<tiles;step++)compute_tile(step,std::false_type{});
'''
s=s[:a]+body+s[b:]
out=Path('/tmp/jzk_compact_half.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
