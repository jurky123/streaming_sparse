"""Start the first summary load after the last exact tile has been consumed."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
s=(root.parent/'page_slope_v1/control.cu').read_text()
a=s.index('    for(int step=0;step<tiles;step++)')
b=s.index('        if(step+1<tiles)',a)
s=s[:a]+'''    for(int step=1;step<tiles;step++){
        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
'''+s[b:]
s=s.replace('        if(!step)continue;\n','')
key='    float* E=reinterpret_cast<float*>(pool+2656);'
s=s.replace(key,'''    // All exact-page shared reads precede the CTA barrier above.
    constexpr int words=1328;
    const uint4* first=reinterpret_cast<const uint4*>(summaries)+(h*((p+15)/16)+1)*words;
    for(int i=threadIdx.x;i<words;i+=128){
        unsigned addr=__cvta_generic_to_shared(pool+words+i);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(first+i));
    }
    asm volatile("cp.async.commit_group;":::"memory");
'''+key)
key='if(valid)tailops::tail(Q,summaries,nullptr,nullptr,ids,O,basis,E,L,pool,p,p,count,mask);'
s=s.replace(key,key+'\n    else asm volatile("cp.async.wait_group 0;":::"memory");')
out=Path('/tmp/jzk_slope_cross_phase.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
