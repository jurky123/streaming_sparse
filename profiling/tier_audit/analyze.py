import json
import sys
from pathlib import Path
from statistics import median
import numpy as np
out=Path(__file__).resolve().parent
policy_dir=out.parents[1]/'reproduction/vbench_current'
sys.path.insert(0,str(policy_dir))
from quota_counts import counts
policy=np.load(policy_dir/'policies.npz')['10_window_step_layer_head']
def read(name):
    return json.loads((out/(name+'.json')).read_text())
def kernel_ms(data,prefix):
    return sum(t/1000 for k,(n,t) in data['kernels'].items() if k.startswith(prefix))
roof=read('roof')
bf,bw=roof.values()
dense=read('dense')
wall=median(dense['times'][1:])
print('dense',wall,'roof',roof)
for method in ('base','intersection','step'):
    x=read(method)
    dense_flops=sparse_flops=route_flops=0
    ws=(216,288,360,432,432)
    for i,w in enumerate(ws):
        pages={'base':w*15//100,'step':w//10,'intersection':24+np.asarray(counts(w)).sum(1)[policy[i]].mean()}[method]
        qh=4608*12
        calls=116
        dense_flops+=calls*qh*4*128*64*w
        sparse_flops+=calls*qh*(6*128*64*pages+w*(8*128+4*128+12*8)+2*8*128)
        route_flops+=calls*12*72*16*2*128*4*w
    measured=kernel_ms(x,'exact(' if method=='base' else '_tail')
    replaced=kernel_ms(dense,'void pytorch_flash::flash_fwd_kernel')-kernel_ms(x,'void pytorch_flash::flash_fwd_kernel')
    def show(label,actual,floor):
        print(label,round(actual,2),round(floor,2),round(actual/floor,2))
    print(method, 'wall',median(x['times'][1:]),'speed',wall/median(x['times'][1:]),'dense and sparse TF',dense_flops/1e12,sparse_flops/1e12,'dense TF/s',dense_flops/replaced/1e9,'sparse TF/s',sparse_flops/measured/1e9,'replaced ms',replaced,'core ms',measured)
    show('fused vs dense throughput',measured,replaced*sparse_flops/dense_flops)
    show('fused vs GEMM',measured,sparse_flops/bf/1e9)
    for k,floor in [('route_and_correction',route_flops/bf/1e9),('summary_and_encoding',580*4*4608*1536/bw/1e6),('clean_commit',203*4*4608*1536/bw/1e6)]:
        show(k,x['stages'][k]['milliseconds'],floor)
