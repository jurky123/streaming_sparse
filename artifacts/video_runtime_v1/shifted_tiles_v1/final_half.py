"""Keep tile order, generate a half-width final QK on P216 and P360."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
s=(root.parents[2]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
s=s.replace('constexpr bool first=decltype(shape)::value;', '''constexpr bool first=decltype(shape)::value==1;
        constexpr bool last=decltype(shape)::value==2;''')
s=s.replace('n=first ? 4 : 0;n<8;n+=2','n=first ? 4 : 0;n<(last ? 4 : 8);n+=2')
s=s.replace('float v=(first && n<4) ||','float v=(first && n<4) || (last && n>=4) ||')
s=s.replace('step<tiles;step++)compute_tile(step,std::false_type{});','step<tiles-1;step++)compute_tile(step,std::false_type{});\n    compute_tile(tiles-1,std::integral_constant<int,2>{});')
out=Path('/tmp/jzk_final_half.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
