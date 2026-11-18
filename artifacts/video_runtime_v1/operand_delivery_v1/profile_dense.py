"""Profile matched P432 Dense and accepted CUDA operation on the same card."""
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from capture import implementation, k, q, v

current,_,flags=implementation(library())
assert not flags.any()
def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
for _ in range(25):
    dense()
    current()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
dense()
current()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
