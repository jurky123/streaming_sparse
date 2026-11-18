"""Build all mask words from the same route registers used for the K/V offsets."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('    // Padded positions')
b=src.index('    const int* list=',a)
src=src[:a]+src[b:]
a=src.index('    int first_offset=')
b=src.index('    A q[8];',a)
src=src[:a]+'''    int first_id=list[l];
    int second_id=l+32<count ? list[l+32] : -1;
    int first_offset=(first_id*12+h)*1024;
    int second_offset=(second_id*12+h)*1024;
    // Each warp publishes four mask words. Every lane contributes up to two IDs.
    #pragma unroll
    for(int j=0;j<4;j++){
        int word=w*4+j;
        unsigned bits=first_id/32==word ? 1u<<(first_id&31) : 0u;
        if(second_id>=0 && second_id/32==word)bits|=1u<<(second_id&31);
        unsigned selected=__reduce_or_sync(0xffffffff,bits);
        if(l==0)mask[word]=selected | ((word==p/32 && p%32) ? (~0u<<(p%32)) : 0u);
    }
'''+src[b:]
a=src.index('        // The first page barrier precedes')
b=src.index('        Slot& kv=',a)
src=src[:a]+src[b:]
path=Path('/tmp/jzk_route_words.cu')
path.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
