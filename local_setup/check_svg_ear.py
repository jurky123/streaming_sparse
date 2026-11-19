import sys
sys.path[:0]=['/tmp/svg_flashinfer_pinned','/home/shuchenweng/jzk/streaming_sparse/external_baselines/Sparse-VideoGen']
import flashinfer,torch
print('flashinfer',flashinfer.__version__,hasattr(flashinfer.sparse,'VariableBlockSparseAttentionWrapper'),flush=True)
from svg.kmeans_utils import dynamic_block_sparse_prune_fwd_flashinfer
q=torch.randn(1,2,128,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(1,2,256,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
kc=k.reshape(1,2,4,64,128).mean(3);vc=v.reshape(1,2,4,64,128).mean(3)
mask=torch.ones(1,2,2,4,device='cuda',dtype=torch.bool);qs=torch.full((1,2,2),64,device='cuda',dtype=torch.int32);ks=torch.full((1,2,4),64,device='cuda',dtype=torch.int32)
y=dynamic_block_sparse_prune_fwd_flashinfer(q,k,v,kc,vc,mask,qs,ks,is_cpu=False)
ref=torch.nn.functional.scaled_dot_product_attention(q,k,v)
print('max_abs_error',(y-ref).abs().max().item(),flush=True)
assert (y-ref).abs().max()<.05
