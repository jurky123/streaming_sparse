"""Full-resolution, uncompressed RGB comparison of two optimization configurations."""
import argparse,json,math,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .weights import ROOT
from ..stream_eval_common import save_video

@torch.no_grad()
def run(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    left,right=Path(a.conservative),Path(a.all_optimized);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lm=json.loads((left/'result.json').read_text());rm=json.loads((right/'result.json').read_text())
    for key in ('video','chunks','history_pages','resolution','fps','frames'):assert lm[key]==rm[key],key
    for key in ('summary_sha','reader_sha','phi_sha','B16_sha','base_sha'):assert lm['weights'][key]==rm['weights'][key],key
    x=np.load(left/'pixels.npy',mmap_mode='r');y=np.load(right/'pixels.npy',mmap_mode='r')
    assert x.shape==y.shape and len(x)==lm['frames'] and x.dtype==y.dtype==np.uint8
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    import lpips
    metric=lpips.LPIPS(net='alex').cuda().eval()
    v=torch.arange(-5,6,device='cuda',dtype=torch.float32);w=torch.exp(-v.square()/(2*1.5**2));w=w/w.sum()
    def blur(z):
        c=z.shape[1];z=F.conv2d(z,w[None,None,None,:].expand(c,1,1,11),groups=c)
        return F.conv2d(z,w[None,None,:,None].expand(c,1,11,1),groups=c)
    per=[]
    for start in range(0,len(x),2):
        u=torch.from_numpy(x[start:start+2].copy()).cuda().permute(0,3,1,2).float()/255
        v=torch.from_numpy(y[start:start+2].copy()).cuda().permute(0,3,1,2).float()/255
        mu,mv=blur(u),blur(v);vu=(blur(u*u)-mu*mu).clamp_min(0);vv=(blur(v*v)-mv*mv).clamp_min(0);cov=blur(u*v)-mu*mv
        ss=((2*mu*mv+.01**2)*(2*cov+.03**2))/((mu*mu+mv*mv+.01**2)*(vu+vv+.03**2))
        percept=metric(2*u-1,2*v-1).flatten().cpu().tolist()
        d=(u-v)*255;mae=d.abs().mean((2,3)).cpu().tolist();mse=d.square().mean((2,3)).cpu().tolist();maximum=d.abs().flatten(1).amax(1).cpu().tolist();ssim=ss.mean((1,2,3)).cpu().tolist()
        for j in range(len(u)):
            if not any(mae[j]):assert abs(ssim[j]-1)<1e-4, 'SSIM identity failed'
            per.append(dict(frame=start+j,lpips=percept[j],ssim=ssim[j],rgb_mae_channels=mae[j],rgb_mse_channels=mse[j],rgb_max_abs=maximum[j]))
    n=len(per);windows={'all':[0,n],'after_initial_21_frames':[21,n]}
    if n==240:windows.update(early=[0,80],middle=[80,160],late=[160,240])
    def summarize(lo,hi):
        r=per[lo:hi];mc=np.mean([z['rgb_mse_channels'] for z in r],axis=0);ac=np.mean([z['rgb_mae_channels'] for z in r],axis=0);m=float(mc.mean())
        return dict(frames=[lo,hi],psnr_db=None if m==0 else 10*math.log10(255**2/m),ssim=float(np.mean([z['ssim'] for z in r])),lpips=float(np.mean([z['lpips'] for z in r])),rgb_mae=float(ac.mean()),rgb_rmse=math.sqrt(m),rgb_max_abs=max(z['rgb_max_abs'] for z in r),rgb_mae_channels=ac.tolist(),rgb_rmse_channels=np.sqrt(mc).tolist())
    result=dict(conservative=str(left),all_optimized=str(right),weights=lm['weights'],shape=list(x.shape),definitions=dict(input='uncompressed decoded uint8 RGB before MP4 encoding',resolution='full 768x512 for all metrics',psnr='10log10(255^2 / mean RGB squared error over window)',ssim='11x11 Gaussian sigma1.5, population covariance, valid border, mean channels and frames',lpips='AlexNet, inputs [-1,1], mean frames',rgb='0..255 units; channel order R,G,B',initial_frames='also report after frame21 to avoid diluting with unchanged initial frames'),windows={k:summarize(*v) for k,v in windows.items()},per_frame=per,acceptance='recorded, not automatically accepted')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    save_video(out/'conservative_vs_all.mp4',(np.concatenate((x[i:i+8],y[i:i+8]),axis=2) for i in range(0,n,8)))
    print(json.dumps(result['windows']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--conservative',required=True);p.add_argument('--all-optimized',required=True);p.add_argument('--output',required=True);run(p.parse_args())
