"""Use a pair of fixed shared slots and split one mask word over adjacent tiles."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
s=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=s.index('    auto compute_tile=')
b=s.index('    // The opposite summary slot',a)
body=s[a:b].replace('int step, auto shape','int step, auto shape, auto parity, unsigned aligned_mask')
body=body.replace('        constexpr bool first=', '        constexpr int odd=decltype(parity)::value;\n        constexpr bool first=')
body=body.replace('slab[1-(next%2)]','slab[odd]').replace('slab[1-(step%2)]','slab[1-odd]')
body=body.replace('        int pg=16*step;\n','').replace('mask[pg/32]>>((pg%32)+2*z)','aligned_mask>>(16*odd)')
i=body.index('    compute_tile(1,')
body=body[:i]+'''    compute_tile(1,std::true_type{},std::true_type{},mask[0]>>(2*z));
    for(int step=2;step<tiles;step+=2){
        unsigned aligned=mask[step/2]>>(2*z);
        compute_tile(step,std::false_type{},std::false_type{},aligned);
        if(step+1<tiles)compute_tile(step+1,std::false_type{},std::true_type{},aligned);
    }
'''
out=Path('/tmp/jzk_page_cursor.cu');out.write_text(s[:a]+body+s[b:])
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
