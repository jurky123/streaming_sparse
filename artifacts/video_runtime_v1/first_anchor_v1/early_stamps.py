"""Record arrival of thread zero at each iteration boundary in the exact page loop."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
source=Path('/tmp/jzk_first_early.cu').read_text()
old='    {\n        const int offset=(list[0]*12+h)*1024;'
assert source.count(old)==1
source=source.replace(old,'    unsigned long long* times=reinterpret_cast<unsigned long long*>(bad+864)+(h*72+t)*(count+2);\n    if(!threadIdx.x)times[0]=clock64();\n'+old)
old='    for(int step=0;step<count;step++){'
assert source.count(old)==1
source=source.replace(old,old+'\n        if(!threadIdx.x)times[step+1]=clock64();')
old='    __syncthreads();\n    float* E='
assert source.count(old)==1
source=source.replace(old,'    if(!threadIdx.x)times[count+1]=clock64();\n'+old)
path=Path('/tmp/jzk_first_earlystamps.cu')
path.write_text(source)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
