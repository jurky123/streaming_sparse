"""Move selected-page index loads before the SM80 exact attention loop."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
base=(Path(__file__).parent/'control.cu').read_text()
for name,load,index in [
    ('lane_ids','int first=list[l], second=l+32<count?list[l+32]:0;', '__shfl_sync(0xffffffff,(step+1)<32?first:second,(step+1)%32)'),
    ('shared_ids','__shared__ int route[64]; if(threadIdx.x<count)route[threadIdx.x]=list[threadIdx.x]; __syncthreads();','route[step+1]')]:
    s=base.replace('    const int* list=ids+(h*72+t)*count;','    const int* list=ids+(h*72+t)*count;\n    '+load)
    s=s.replace('int page=list[step+1];','int page='+index+';')
    out=root/(name+'.cu')
    out.write_text(s)
    with (root/(name+'.log')).open('w') as f:
        subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-lineinfo','-Xptxas=-v','-shared','-Xcompiler=-fPIC',str(out),'-o','/tmp/jzk_'+name+'.so'],stderr=f,check=True)
