import ctypes as c,statistics,torch
from local_experiments.video_runtime_v1.ffn_rounded_gelu import gelu_table,library
lib=c.CDLL('/tmp/ffn_tiles.so')
lib.ffn_gelu.argtypes=library().ffn_gelu.argtypes
lib.set_gelu_table.argtypes=[c.c_void_p]
x=torch.randn(4608,1536,device='cuda',dtype=torch.bfloat16)
w=torch.randn(8960,1536,device='cuda',dtype=x.dtype)/1536**.5
bias=torch.randn(8960,device='cuda',dtype=x.dtype)
y=x.new_empty(4608,8960)
table=gelu_table(x.device)
assert lib.set_gelu_table(table.data_ptr())==0
def run(i):
    return lib.ffn_gelu(i,4608,8960,1536,x.data_ptr(),w.data_ptr(),bias.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
stats={}
for i in range(9):
    if run(i):continue
    if i==0:ref=y.clone()
    stats[i]=[]
    print('tile',i,'diff',(y!=ref).float().mean().item(),flush=True)
for rep in range(7):
    for i in (list(stats) if rep%2==0 else reversed(stats)):
        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(15):run(i)
        end.record();end.synchronize()
        stats[i].append(start.elapsed_time(end)/15)
print({i:statistics.median(v) for i,v in stats.items()})
