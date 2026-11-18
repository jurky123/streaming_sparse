"""Probe exact BF16 basis expansion and paired output stores after the tail loop."""
from pathlib import Path
import subprocess
import sys
name=sys.argv[1]
control=Path('/tmp/jzk_tailmax_direct_mass.cu')
if not control.exists():
    subprocess.run([sys.executable,str(Path(__file__).with_name('build.py')),'direct_mass'],check=True)
s=control.read_text()
if name=='basis_bits':
    a=s.index('        B b={{tf(__bfloat162float(basis')
    b=s.index('        C proj=',a)
    s=s[:a]+'''        unsigned values=packed(basis+(h*128+n*8+g)*8+2*z);
        B b={{values<<16,values&0xffff0000u}};
'''+s[b:]
elif name in ('pair_store','vector_store'):
    a=s.index('        #pragma unroll\n        for(int r=0;r<2;r++)',s.index('        C proj={}'))
    b=s.index('\n    }',a)
    s=s[:a]+'''        #pragma unroll
        for(int r=0;r<2;r++){
            int dim=n*8+2*z,local=(row+8*r)%64;
            float result[2];
            #pragma unroll
            for(int j=0;j<2;j++){
                float tv=divide(am[n].v[2*r+j],inverse[r])+proj.v[2*r+j];
                float e=exact[local*128+dim+j];
                result[j]=e*(1.f-rho[r])+tv*rho[r];
            }
            *reinterpret_cast<unsigned*>(out+((row+8*r)*12+h)*128+dim)=pair(result[0],result[1]);
        }'''+s[b:]
    if name=='vector_store':
        a=s.index('    A high,low;')
        s=s[:a]+"""    // The opposite summary slot is free after the final tile's CTA barrier.
    unsigned* stage_words=reinterpret_cast<unsigned*>(slab[tiles%2]);
"""+s[a:]
        s=s.replace('*reinterpret_cast<unsigned*>(out+((row+8*r)*12+h)*128+dim)=pair(result[0],result[1]);',
                    'stage_words[local*64+(dim/2 ^ ((local&7)*4))]=pair(result[0],result[1]);')
        i=s.index('\n}\n\n}\n__global__ void exact')
        s=s[:i]+"""    __syncwarp();
    #pragma unroll
    for(int i=l;i<256;i+=32) {
        int row_in_warp=i/16, vec=i%16, local=w*16+row_in_warp;
        uint4 data=*reinterpret_cast<uint4*>(stage_words+local*64+((vec*4)^((local&7)*4)));
        reinterpret_cast<uint4*>(out+((t*64+local)*12+h)*128)[vec]=data;
    }
"""+s[i:]
else:
    raise ValueError(name)
p=Path('/tmp/jzk_tailmax_'+name+'.cu')
p.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(p),'-o',str(p.with_suffix('.so'))],check=True)
