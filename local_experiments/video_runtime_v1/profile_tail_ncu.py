"""A single real P432 tail event for hardware-counter inspection."""
import torch,triton as tr
from .weights import ROOT
from ..review_v1.tuned_attention import _tail
from ..high_sparse_runtime import _budget_mask
@torch.no_grad()
def main():
 torch.set_num_threads(4);s=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True);q=s['q'].cuda();p=432;n=4608;h=12;d=128
 keys,mean,detail=[s['summary'][k].cuda().contiguous() for k in ('keys','mean','detail')];ids=s['ids'].cuda().contiguous()[None]
 bits=torch.empty(h,72,16,device='cuda',dtype=torch.uint32);_budget_mask[(72,h)](ids,bits,72,43,64,num_warps=4)
 mu=torch.empty(h,n,d,device='cuda');code=torch.empty(h,n,8,device='cuda');lse=torch.empty(h,n,device='cuda')
 for _ in range(2):
  _tail[(72,h)](q,keys,mean,detail,bits,mu,code,lse,p,n,h,d,BQ=64,BR=64,BF16_AV=True,num_warps=4,num_stages=2,enable_fp_fusion=False);torch.cuda.synchronize()
if __name__=='__main__':main()
