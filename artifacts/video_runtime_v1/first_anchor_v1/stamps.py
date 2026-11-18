"""Record arrival of thread zero at each iteration boundary in the exact page loop."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
source=(Path(__file__).with_name('control.cu')).read_text()
old='    for(int step=-1;step<count;step++){'
new='''    unsigned long long* times=reinterpret_cast<unsigned long long*>(bad+864)+(h*72+t)*(count+2);
    for(int step=-1;step<count;step++){
        if(!threadIdx.x)times[step+1]=clock64();'''
assert source.count(old)==1
source=source.replace(old,new)
old='    __syncthreads();\n    float* E='
assert source.count(old)==1
source=source.replace(old,'    if(!threadIdx.x)times[count+1]=clock64();\n'+old)
path=Path('/tmp/jzk_first_stamps.cu')
path.write_text(source)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
