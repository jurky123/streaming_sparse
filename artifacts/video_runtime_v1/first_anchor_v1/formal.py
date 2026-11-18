"""Share the identical K/V transfer code in the prologue and the normal page loop."""
from pathlib import Path
import subprocess

source=Path('/tmp/jzk_first_early.cu').read_text()
a=source.index('            const uint4* fk=')
b=source.index('            }\n',a)+len('            }\n')
body=source[a:b]
assert source.count(body)==2
call='            copy_page(slots[slot], K, V, offset);\n'
source=source.replace(body,call)
helper='''__device__ __forceinline__ void copy_page(uint4 (&target)[2][1024],
                                          const bf* K,const bf* V,int offset) {
'''+''.join(line[8:] for line in body.replace('slots[slot]', 'target').splitlines(keepends=True))+'}\n'
i=source.index('__global__ void exact(')
source=source[:i]+helper+source[i:]
a=source.index('    {\n        const int offset=(list[0]')
b=source.index('    // A Q64 selects',a)
source=source[:a]+'''    // Issue the own-page transfer before loading Q and the remaining page offsets.
    copy_page(slots[count&1], K, V, (list[0]*12+h)*1024);
    asm volatile("cp.async.commit_group;":::"memory");
'''+source[b:]
source=source.replace('''        {
            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
        }
''','''        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
''')
path=Path('/tmp/jzk_first_formal.cu')
path.write_text(source)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',
                str(path),'-o',str(path.with_suffix('.so'))],check=True)
