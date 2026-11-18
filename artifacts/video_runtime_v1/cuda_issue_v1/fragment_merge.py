"""Keep exact outputs in lane order in shared memory through the compensation stage."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
s=(root.parent/'cuda_base_layout_v1/fixed15_attention.cu').read_text()
a=s.index('        for(int r=0;r<2;r++)\n            #pragma unroll',s.index('C proj={};project'))
b=s.index('\n    }',a)
s=s[:a]+'''        for(int r=0;r<2;r++){
            int ix=(r*16+n)*128+threadIdx.x;
            float2 e=reinterpret_cast<const float2*>(exact)[ix];
            int dim=n*8+2*z;
            float a=e.x*(1.f-rho[r])+(divide(am[n].v[2*r],inverse[r])+proj.v[2*r])*rho[r];
            float b=e.y*(1.f-rho[r])+(divide(am[n].v[2*r+1],inverse[r])+proj.v[2*r+1])*rho[r];
            *reinterpret_cast<unsigned*>(out+((row+8*r)*12+h)*128+dim)=pair(a,b);
        }'''+s[b:]
a=s.index('        for(int n=0;n<16;n++)\n            #pragma unroll',s.index('if(pair==0)L[idx]'))
b=s.index('\n    }',a)
s=s[:a]+'''        for(int n=0;n<16;n++){
            float2 normalized;
            asm("div.full.f32 %0,%1,%2;":"=f"(normalized.x):"f"(out[n].x[2*r]),"f"(s));
            asm("div.full.f32 %0,%1,%2;":"=f"(normalized.y):"f"(out[n].x[2*r+1]),"f"(s));
            reinterpret_cast<float2*>(E)[(r*16+n)*128+threadIdx.x]=normalized;
        }'''+s[b:]
out=root/'merge.cu';out.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_80','-Xptxas=-v','-shared','-Xcompiler=-fPIC',str(out),'-o','/tmp/jzk_merge.so'],check=True)
