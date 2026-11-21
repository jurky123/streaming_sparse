import json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from local_experiments.stream_eval_common import native_vae,decode
here=Path(__file__).resolve().parent
@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    vae=native_vae();metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    for f in sorted(here.glob('*.pt')):
        out=f.with_suffix('.quality.json')
        if out.exists():continue
        n=int(f.stem.rsplit('_',1)[1]);frames=80 if n==7 else 480
        ref=np.load(here.parent/'clean_forward'/('dense_before_5s.npy' if n==7 else 'dense_before.npy'),mmap_mode='r')
        z=torch.load(f,weights_only=True).cuda()
        rgb=np.concatenate(decode(vae,z,frames))
        accum=[]
        for i in range(0,frames,4):
            def batch(a):return torch.from_numpy(a[i:i+4].copy()).cuda().permute(0,3,1,2).float()/255
            x,y=batch(rgb),batch(ref)
            mse=(x-y).square().mean((1,2,3))
            x,y=[F.interpolate(a,(256,384),mode='bilinear',align_corners=False) for a in (x,y)]
            score=metric(x*2-1,y*2-1).flatten()
            def pool(t):return F.avg_pool2d(t,11,1,5,count_include_pad=False)
            mx,my=pool(x),pool(y)
            vx=(pool(x*x)-mx*mx).clamp_min(0);vy=(pool(y*y)-my*my).clamp_min(0)
            ss=((2*mx*my+.01**2)*(2*(pool(x*y)-mx*my)+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
            accum.append(torch.stack((mse,score,ss.mean((1,2,3))),-1).cpu())
        mse,perceptual,ssim=torch.cat(accum).double().mean(0).tolist()
        data=dict(psnr=-10*np.log10(mse),lpips=perceptual,ssim=ssim)
        out.write_text(json.dumps(data));print(f.name,data,flush=True)
main()
