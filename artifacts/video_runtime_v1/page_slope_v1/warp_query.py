"""Load native Q in aligned vectors using a private tile in the inactive shared slot."""
from pathlib import Path
import subprocess
import sys

name=sys.argv[1]
root=Path(__file__).parent
s=(root/'control.cu').read_text()
if name!='control':
    assert name in ('exact_query','tail_query','both_query')
    for phase in ('exact','tail'):
        if name not in (phase+'_query','both_query'):
            continue
        a=s.index('    #pragma unroll\n    for(int k=0;k<8;k++)',s.index('    A q[8];',s.index('__global__ void exact' if phase=='exact' else 'void tail(')))
        b=s.index('    Four out[16]' if phase=='exact' else '    // One slab',a)
        pool='pool+2048' if phase=='exact' else 'sm'
        member='x' if phase=='exact' else 'v'
        body=f"""    uint4* qtile={pool}+w*256;
    #pragma unroll
    for(int i=l;i<256;i+=32){{
        int qr=i/16, vec=i%16;
        qtile[qr*16+(vec^(qr&7))]=reinterpret_cast<const uint4*>(Q+((t*64+w*16+qr)*12+h)*128)[vec];
    }}
    __syncwarp();
    #pragma unroll
    for(int k=0;k<8;k++){{
        int qr=((l/8)%2)*8+(l%8);
        int vec=2*k+l/16;
        unsigned address=__cvta_generic_to_shared(qtile+qr*16+(vec^(qr&7)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}},[%4];"
            :"=r"(q[k].{member}[0]),"=r"(q[k].{member}[1]),"=r"(q[k].{member}[2]),"=r"(q[k].{member}[3]):"r"(address));
    }}
"""
        s=s[:a]+body+s[b:]
out=Path('/tmp/jzk_slope_'+name+'.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
