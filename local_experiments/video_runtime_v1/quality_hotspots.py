"""Decode the single-video numerical ablation and compare actual RGB outputs."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .weights import ROOT
from ..stream_eval_common import native_vae,decode,save_video
from ..learned_cache.io import atomic

@torch.no_grad()
def run(a):
    torch.set_num_threads(4);root=ROOT/'artifacts/video_runtime_v1/hotspot_quality';root.mkdir(parents=True,exist_ok=True)
    paths=dict(dense=root/'dense/latents.pt',original=ROOT/'artifacts/video_runtime_v1/hotspot_round3/e2e_before/latents.pt',tail=ROOT/'artifacts/video_runtime_v1/hotspot_round3/e2e_tail/latents.pt')
    if (root/'trained/latents.pt').exists():paths['trained']=root/'trained/latents.pt'
    if a.mode=='decode':
        vae=native_vae()
        for name in a.methods:
            dest=root/name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'pixels.npy').exists():continue
            start=time.perf_counter();frames=decode(vae,torch.load(paths[name],weights_only=True).cuda(),105);torch.cuda.synchronize();seconds=time.perf_counter()-start
            np.save(dest/'pixels.npy',np.concatenate(frames));save_video(dest/'video.mp4',frames);atomic(dest/'decode.json',dict(seconds=seconds,latent=str(paths[name]),frames=105));print(name,seconds,flush=True)
    else:
        import lpips
        from ..high_sparse_experiment import prefix_quality
        os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache');metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        pixels={name:np.load(root/name/'pixels.npy',mmap_mode='r') for name in paths};rows={}
        for left,right in [('original','dense'),('tail','dense'),('tail','original')]+([('trained','dense'),('trained','tail')] if 'trained' in paths else []):
            whole=prefix_quality([pixels[left]],[pixels[right]],metric);windows={'all':whole}
            for label,start,stop in [('early',21,47),('middle',47,74),('late',74,101)]:
                v={k:values[start-21:stop-21] for k,values in whole['per_frame'].items()};m=float(np.mean(v['mse']))
                windows[label]=dict(lpips=float(np.mean(v['lpips'])),ssim=float(np.mean(v['ssim'])),psnr=100. if m==0 else -10*np.log10(m),window=[start,stop],per_frame=v)
            rows[f'{left}_vs_{right}']=windows
        atomic(root/'quality.json',rows)
        save_video(root/'comparison.mp4',(np.stack([np.concatenate([pixels[n][i] for n in ('dense','original','tail')],axis=1) for i in range(j,min(j+8,105))]) for j in range(0,105,8)))
        from PIL import Image,ImageDraw
        frames=[24,48,72,96];canvas=Image.new('RGB',(768*3,512*len(frames)+32),'white');draw=ImageDraw.Draw(canvas)
        for col,name in enumerate(('dense','original','tail')):draw.text((col*768+10,8),name,fill='black')
        for row,i in enumerate(frames):
            for col,name in enumerate(('dense','original','tail')):canvas.paste(Image.fromarray(pixels[name][i]),(col*768,32+row*512))
        canvas.save(root/'contact_sheet.jpg');print(json.dumps({k:{w:{m:v for m,v in q.items() if m!='per_frame'} for w,q in windows.items()} for k,windows in rows.items()}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['decode','metrics'],required=True);p.add_argument('--methods',nargs='+',default=['dense','original','tail']);run(p.parse_args())
