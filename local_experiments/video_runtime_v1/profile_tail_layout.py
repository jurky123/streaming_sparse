"""Profile saved layout-only TTGIR independently of compiler warmup."""
import argparse
import torch,triton
from .check_tail_layout import compiler
from .weights import ROOT
from ..high_sparse_runtime import _budget_mask
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--variant',default='blocked',choices=['blocked','paired','phase1']);variant=parser.parse_args().variant
 torch.set_num_threads(4);root=ROOT/'artifacts/video_runtime_v1/tail_layout_v1';s=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
 q=s['q'].cuda();keys,mean,detail=[s['summary'][k].cuda().contiguous() for k in ('keys','mean','detail')];ids=s['ids'].cuda().contiguous()[None]
 bits=torch.empty(12,72,16,device='cuda',dtype=torch.uint32);_budget_mask[(72,12)](ids,bits,72,43,64,num_warps=4)
 mu=torch.empty(12,4608,128,device='cuda');code=torch.empty(12,4608,8,device='cuda');lse=torch.empty(12,4608,device='cuda')
 kernel=triton.compile(str(root/(variant+'.ttgir')),options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False))
 for _ in range(2):kernel[(72,12,1)](q,keys,mean,detail,bits,mu,code,lse);torch.cuda.synchronize()
if __name__=='__main__':main()
