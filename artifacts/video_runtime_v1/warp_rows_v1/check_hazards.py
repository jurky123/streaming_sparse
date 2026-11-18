"""Exercise one CTA through the complete fused loop under Compute Sanitizer."""
import sys
from measure import folder,ROOT,torch,triton
mode=sys.argv[1]
data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
q,k,v=[data[key].cuda().contiguous() for key in ('q','k','v')]
keys,mean,detail=[data['summary'][key].cuda().contiguous() for key in ('keys','mean','detail')]
ids=data['ids'][...,:43].cuda().contiguous()
basis=data['basis'].cuda().bfloat16().contiguous()
out=torch.empty_like(q)
compiled=triton.compile(str(folder/f'{mode}_432.ttgir'),options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False))
args=(q,keys,mean,detail,basis,out,k,v,ids)
if '--profile' in sys.argv:
    for _ in range(50):
        compiled[(72,12,1)](*args)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    compiled[(72,12,1)](*args)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
else:
    compiled[(1,1,1)](*args)
torch.cuda.synchronize()
