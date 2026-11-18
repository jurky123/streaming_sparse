"""Start the first page copy from list[0] before loading Q or the later-page offsets."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(Path(__file__).with_name('control.cu')).read_text()
a=src.index('            int slot=(step+1+count)&1;')
b=src.index('        }else{',a)
issue=src[a:b].replace('(step+1+count)&1','count&1')
i=src.index('    // A Q64 selects')
src=src[:i]+'''    {
        const int offset=(list[0]*12+h)*1024;
'''+issue+'''        asm volatile("cp.async.commit_group;":::"memory");
    }
'''+src[i:]
src=src.replace('for(int step=-1;step<count;step++){','for(int step=0;step<count;step++){').replace('        if(step>=0) {','        {').replace('        if(step<0)continue;','')
path=Path('/tmp/jzk_first_early.cu')
path.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
