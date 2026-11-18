"""Specialize the last full arithmetic tile only for its prefetch behavior."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
start=src.index('    auto compute_tile=')
end=src.index('    // The opposite summary slot', start)
old=src[start:end]
s=old.replace('int step, auto shape','int step, auto shape, auto more')
s=s.replace('if(step+1<tiles){','if constexpr(decltype(more)::value){')
s=s.replace('        }\n        asm volatile("cp.async.commit_group;":::"memory");','            asm volatile("cp.async.commit_group;":::"memory");\n        }')
s=s.replace('compute_tile(1,std::true_type{});','compute_tile(1,std::true_type{},std::true_type{});')
s=s.replace('step<tiles;step++)compute_tile(step,std::false_type{});','step<tiles-1;step++)compute_tile(step,std::false_type{},std::true_type{});\n    compute_tile(tiles-1,std::false_type{},std::false_type{});')
assert s!=old and 'if(step+1<tiles)' not in s
out=Path('/tmp/jzk_page_lastfull.cu')
out.write_text(src[:start]+s+src[end:])
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
