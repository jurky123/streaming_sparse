"""Compare one directed multiply after max with a captured-anchor diagnostic."""
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

root = Path(__file__).resolve().parents[3]
source = (Path(__file__).with_name('control.cu')).read_text()
a = source.index('        if(pg==0) {')
b = source.index('        A high[4]', a)
original = source[a:b]
assignment = 'anchor[r]=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));'
assert original.count(assignment) == 1
scalar = '__fmul_rd(score[n].x[j],log2_scale)'
assert original.count(scalar) == 1
late = original.replace(scalar,'score[n].x[j]').replace(assignment,
    'anchor[r]=__fmul_rd(fmaxf(m,__shfl_xor_sync(0xffffffff,m,1)),log2_scale);')
slot = 'reinterpret_cast<float*>(bad+864)[(h*72+t)*64+(row+8*r)%64]'
capture = original.replace(assignment, assignment + '\n                if(pair==0) '+slot+'=anchor[r];')
oracle = '        if(pg==0){\n            for(int r=0;r<2;r++) anchor[r]='+slot+';\n        }\n'
def build(item):
    name, body = item
    path = Path(f'/tmp/jzk_first_{name}.cu')
    path.write_text(source[:a]+body+source[b:])
    subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',
                    str(path),'-o',str(path.with_suffix('.so'))],check=True)
with ThreadPoolExecutor(3) as pool:
    list(pool.map(build,[('late',late),('capture',capture),('oracle',oracle)]))
