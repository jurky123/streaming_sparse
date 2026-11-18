"""Record arrival of thread zero at each iteration boundary in the exact page loop."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
source=(Path(__file__).with_name('control.cu')).read_text()
def mark(needle, before, condition, slot):
    global source
    assert source.count(needle)==1, needle
    record=f'if(({condition}) && !threadIdx.x) times[{slot}]=clock64();'
    source=source.replace(needle,(record+'\n        '+needle) if before else (needle+'\n        '+record))
loop='    for(int step=-1;step<count;step++){'
source=source.replace(loop, '    unsigned long long* times=reinterpret_cast<unsigned long long*>(bad+864)+(h*72+t)*6;\n'+loop)
mark('int pg=step;',False,'step==0',0)
mark('if(step+1<count){',True,'step==0',1)
mark('Slot& kv=slots[(step+count)&1];',False,'step==0',2)
mark('if(pg==0) {',False,'true',3)
mark('A high[4],low[4];',True,'step==0',4)
mark('int pg=step;',False,'step==1',5)
path=Path('/tmp/jzk_first_phases.cu')
path.write_text(source)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
