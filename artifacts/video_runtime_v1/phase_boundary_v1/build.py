"""Build single-factor changes at the exact-to-compensation boundary."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).parent
name=sys.argv[1]
src=(root/'control.cu').read_text()
if name=='early_summary':
    # Tile zero lies within the guaranteed 24-page first-frame anchor.
    a=src.index('    for(int step=0;step<tiles;step++){')
    b=src.index('        if(step+1<tiles)',a)
    src=src[:a]+src[a:b].replace('step=0','step=1').replace('if(step){','if(step>1){')+src[b:]
    src=src.replace('        if(!step)continue;','')
    pos=src.index('    float* E=')
    src=src[:pos]+'''    constexpr int summary_words=1328;
    int tiles=(p+15)/16;
    const uint4* first_summary=reinterpret_cast<const uint4*>(summaries)+(h*tiles+1)*summary_words;
    for(int i=threadIdx.x;i<summary_words;i+=128){
        unsigned addr=__cvta_generic_to_shared(pool+summary_words+i);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(first_summary+i));
    }
    asm volatile("cp.async.commit_group;":::"memory");
'''+src[pos:]
    src=src.replace('    int valid=','    asm volatile("cp.async.wait_group 0;":::"memory");\n    int valid=')
elif name in ('head_pages','group8_pages'):
    for idx in ('list[l]','list[l+32]'):
        layout=(f'(h*432+{idx})' if name=='head_pages' else
                f'(({idx}/8*12+h)*8+{idx}%8)')
        src=src.replace(f'({idx}*12+h)*1024',layout+'*1024')
elif name=='bounded_div':
    # Denominators in exact (valid tiles) and safe tail normalization stay
    # within the regular exponent range. Keep the independent rho division.
    src=src.replace('div.full.f32 %0,%1,%2;":"=f"(z)','div.approx.ftz.f32 %0,%1,%2;":"=f"(z)')
    src=src.replace('float v=divide(ac.v[2*r+k],safe);','float v; asm("div.approx.ftz.f32 %0,%1,%2;":"=f"(v):"f"(ac.v[2*r+k]),"f"(safe));')
    src=src.replace('float tv=divide(am[n].v[2*r+j],inverse[r])+proj.v[2*r+j];','float normalized; asm("div.approx.ftz.f32 %0,%1,%2;":"=f"(normalized):"f"(am[n].v[2*r+j]),"f"(inverse[r]));\n                float tv=normalized+proj.v[2*r+j];')
else:
    assert name=='control'

out=Path('/tmp/jzk_phase_'+name+'.cu')
out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
