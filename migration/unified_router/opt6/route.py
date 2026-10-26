import torch
import triton as tr
from ..opt5.network import OptimizedRoute,OPTIONS
from .packed import _packed_actions
from .fused import _fused_factor
from .factor import _factor_prepare,_factor_finish

class HeadRoute(OptimizedRoute):
    def __init__(self,method='packed',warps=4,pair='tf32x3',**options):
        super().__init__(**OPTIONS,**options)
        self.head_method=method;self.head_warps=warps;self.pair_precision=pair

    def actions(self,q,sketch,score,mask,rem,add,beta,step,stats,cv,head):
        h,n,b=score.shape;out=torch.empty_like(mask)
        if self.head_method=='fused':
            _fused_factor[(h*n,)](stats,score,cv,sketch.costs,sketch.origins,rem,add,head.w1,head.b1,head.mean,head.std,step,
                mask,head.w2,head.b2,out,n,b,tr.next_power_of_2(b),sketch.history_tokens,sketch.current_tokens,beta,self.pair_precision,
                num_warps=self.head_warps,enable_fp_fusion=False)
            return out
        if self.head_method=='factor':
            tmp=torch.empty((h*n,384),device=q.device,dtype=torch.float32)
            _factor_prepare[(h*n,)](stats,score,cv,sketch.costs,sketch.origins,rem,add,head.w1,head.b1,head.mean,head.std,step,tmp,
                n,b,sketch.history_tokens,sketch.current_tokens,beta,self.pair_precision,num_warps=self.head_warps,enable_fp_fusion=False)
            _factor_finish[(h*n,)](tmp,stats,mask,rem,add,head.w1,head.w2,head.b2,out,n,b,tr.next_power_of_2(b),
                num_warps=self.head_warps,enable_fp_fusion=False)
            return out
        _packed_actions[(h*n,)](stats,score,cv,mask,sketch.costs,sketch.origins,rem,add,
            head.w1,head.b1,head.w2,head.b2,head.mean,head.std,step,out,n,b,tr.next_power_of_2(b),
            sketch.history_tokens,sketch.current_tokens,beta,'fp16',num_warps=self.head_warps,enable_fp_fusion=False)
        return out
