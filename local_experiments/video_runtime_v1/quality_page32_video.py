import os,json
import torch,numpy as np
from .weights import ROOT
from ..high_sparse_experiment import prefix_quality

def main():
    torch.set_num_threads(4)
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    import lpips
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    root=ROOT/'artifacts/video_runtime_v1/reader_page32_v1'
    a=np.load(root/'baseline/pixels.npy',mmap_mode='r');b=np.load(root/'page32/pixels.npy',mmap_mode='r')
    result=prefix_quality([b],[a],metric)
    (root/'quality.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='per_frame'})
if __name__=='__main__':main()
