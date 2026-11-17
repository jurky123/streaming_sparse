"""15/30-second phase timings and decoded video comparisons, one city_walk seed."""
import argparse,json,os,time,statistics
from pathlib import Path
import numpy as np
import torch
from .weights import ROOT
from ..learned_cache.io import atomic
from ..stream_eval_common import native_vae,decode,save_video

@torch.no_grad()
def run(a):
    torch.set_num_threads(4);root=Path(a.root)
    if a.mode=='decode':
        meta=json.loads((root/'result.json').read_text());frames=meta['frames'];z=torch.load(root/'latents.pt',weights_only=True).cuda();vae=native_vae();decode(vae,z,frames);times=[]
        for _ in range(2):
            torch.cuda.synchronize();start=time.perf_counter();pixels=decode(vae,z,frames);torch.cuda.synchronize();times.append(time.perf_counter()-start)
        np.save(root/'pixels.npy',np.concatenate(pixels));save_video(root/'video.mp4',pixels)
        atomic(root/'decode.json',dict(seconds=times,median_seconds=statistics.median(times),generation_seconds=meta['median_seconds'],generation_plus_decode_seconds=meta['median_seconds']+statistics.median(times),timing='sum of separately warmed generation and decode phase medians; file IO/encoding excluded'))
    else:
        import lpips
        from ..high_sparse_experiment import prefix_quality
        os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache');metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        names=['dense','original','optimized'];folders={n:root/('native_dense' if n=='dense' else n) for n in names};pixels={n:np.load(folders[n]/'pixels.npy',mmap_mode='r') for n in names};frames=len(pixels['dense']);quality={};performance={}
        for name in names:
            performance[name]=json.loads((folders[name]/'decode.json').read_text())
            performance[name]['generation_speedup_vs_dense']=performance['dense']['generation_seconds']/performance[name]['generation_seconds']
            performance[name]['generation_vae_speedup_vs_dense']=performance['dense']['generation_plus_decode_seconds']/performance[name]['generation_plus_decode_seconds']
        for left,right in [('original','dense'),('optimized','dense'),('optimized','original')]:
            result={}
            for label,start in ([('all',21)] if frames==105 else [('early',21),('middle',(frames-80)//2),('late',frames-80)]):
                x=np.concatenate((pixels[left][:21],pixels[left][start:start+80]));y=np.concatenate((pixels[right][:21],pixels[right][start:start+80]));q=prefix_quality([x],[y],metric);q['window']=[start,start+80];result[label]=q
            quality[left+'_vs_'+right]=result
        save_video(root/'comparison.mp4',(np.stack([np.concatenate([pixels[n][i] for n in names],axis=1) for i in range(j,min(j+8,frames))]) for j in range(0,frames,8)))
        from PIL import Image,ImageDraw
        ids=[24,48,96] if frames==105 else [40,frames//2,frames-20];sheet=Image.new('RGB',(768*3,512*3+32),'white');draw=ImageDraw.Draw(sheet)
        for col,n in enumerate(names):draw.text((col*768+8,8),n,fill='black')
        for row,i in enumerate(ids):
            for col,n in enumerate(names):sheet.paste(Image.fromarray(pixels[n][i]),(col*768,32+row*512))
        sheet.save(root/'contact_sheet.jpg');atomic(root/'comparison.json',dict(frames=frames,seconds=frames/16,history_pages=360,history_chunks=5,resolution=[512,768],performance=performance,quality=quality,contact_frames=ids))
        print(json.dumps(performance),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',choices=['decode','metrics'],required=True);run(p.parse_args())
