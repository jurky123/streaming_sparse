"""Compile Q32 epilogue resource usage without creating a CUDA context."""
import triton,json,subprocess
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget, AttrsDescriptor
from pathlib import Path
from local_experiments.gap_tail_fused_merge_q32 import _precision_tail
sig=dict(zip(_precision_tail.arg_names[:12],['*bf16','*bf16','*fp32','*u32','*fp32','*fp32','*fp32','*bf16','*fp32','*bf16','*fp32','*fp32']))
const=dict(MODE=1,P=410,N=4608,H=12,D=128,BQ=32,BR=64)
k=triton.compile(ASTSource(_precision_tail,signature=sig,constants=const,attrs=AttrsDescriptor.from_dict(dict(cls='AttrsDescriptor',arg_properties={'tt.divisibility':list(range(12))}))),target=GPUTarget('cuda',90,32),options=dict(num_warps=4,num_stages=2))
r=Path('artifacts/gap_tail_fused_merge_q32_20261101');r.mkdir(exist_ok=True)
f=r/'offline_aligned.cubin';f.write_bytes(k.asm['cubin'])
result=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(f)],text=True)
data=dict(shared=k.metadata.shared,resource_usage=result,scope='Offline compile only; no runtime or numerical validation. All input pointers specialized to16-byte alignment; exact runtime compilation still needs verification.');(r/'offline_aligned_resources.json').write_text(json.dumps(data,indent=2)+'\n');print(data)
