"""Interleave the next selected page's QK with the current page's two-term AV."""
from pathlib import Path
import subprocess

root=Path(__file__).parent
source=(root.parent/'cuda_base_layout_v1/fixed15_attention.cu').read_text()
start=source.index('    for(int step=-1;step<count;step++)')
end=source.index('        #pragma unroll\n        for(int r=0;r<2;r++)',start)
orig=source[start:end]
copy=orig[orig.index('            int page='):orig.index('        }\n        asm volatile("cp.async.commit')]
copy=copy.replace('list[step+1]','list[next]').replace('(step+1)&1','next&1')
qk=orig[orig.index('                uint4 u=reinterpret_cast'):orig.index('            }\n',orig.index('                uint4 u=reinterpret_cast'))]
qk=qk.replace('kv[0]','nextkv[0]')
s=source[:start]+'''    auto copy_page = [&](int next){
'''+copy+'''        asm volatile("cp.async.commit_group;":::"memory");
    };
    copy_page(0);
    asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
    Four score[8]={};
    {
        Slot& nextkv=slots[0];
        #pragma unroll
        for(int k=0;k<8;k++)
            #pragma unroll
            for(int n=0;n<8;n+=2){
'''+qk+'''            }
    }
    for(int step=0;step<count;step++){
        int pg=step;
        if(step+1<count)copy_page(step+1);
        Slot& kv=slots[step&1];
'''+source[end:]
pos=s.index('        #pragma unroll\n        for(int n=0;n<16;n+=2)',start)
s=s[:pos]+'''        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
        Slot& nextkv=slots[(step+1)&1];
        #pragma unroll
        for(int n=0;n<8;n++)score[n]=Four{};
'''+s[pos:]
pos=s.index('            #pragma unroll\n            for(int k=0;k<4;k++) {\n                mma(tmp[0],low',pos)
s=s[:pos]+'''            if(step+1<count){
                int k=n/2;
                #pragma unroll
                for(int n=0;n<8;n+=2){
'''+qk+'''                }
            }
'''+s[pos:]
pos=s.index("    }\n    __syncthreads();\n    float* E",pos)
s=s[:pos]+"        __syncthreads();\n"+s[pos:]
out=root/'issue.cu';out.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_80','--fmad=false','-lineinfo','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o','/tmp/jzk_issue.so'],check=True)
