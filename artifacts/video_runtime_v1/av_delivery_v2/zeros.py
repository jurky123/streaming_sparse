"""Count exact zero high/low MMA A fragments in a real invocation."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('__global__ void exact')
s=src[a:]
s=s.replace('int p,int count) {','int p,int count,int* counts) {',1)
s=s.replace('    Four out[16]', '    int high_zero=0,low_zero=0;\n    Four out[16]',1)
s=s.replace('        // Accumulate both BF16','''        #pragma unroll
        for(int k=0;k<4;k++){
            unsigned a=0,b=0;
            #pragma unroll
            for(int i=0;i<4;i++){a|=high[k].x[i];b|=low[k].x[i];}
            high_zero+=__all_sync(0xffffffff,(a&0x7fff7fff)==0);
            low_zero+=__all_sync(0xffffffff,(b&0x7fff7fff)==0);
        }
        // Accumulate both BF16''',1)
s=s.replace('    float* E=', '''    if(l==0){int at=((h*72+t)*4+w)*2;counts[at]=high_zero;counts[at+1]=low_zero;}
    float* E=''',1)
s=s.replace('int count,cudaStream_t stream){','int count,int* counts,cudaStream_t stream){',1).replace('basis,bad,p,count);','basis,bad,p,count,counts);',1)
out=Path('/tmp/jzk_av_zero_count.cu')
out.write_text(src[:a]+s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
