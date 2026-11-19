import json,sys,subprocess
from pathlib import Path
import torch
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[2]/"eval_env_local/lib/python3.10/site-packages"))
import clip
from torchvision.transforms import Compose,Resize,CenterCrop,InterpolationMode,Normalize

out=Path(__file__).parent
root=out.parent.parent
torch.set_num_threads(2)
model,_=clip.load(str(root/'cache/clip_model/ViT-B-32.pt'),device='cuda')
transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC,antialias=False),CenterCrop(224)])
norm=Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))
@torch.no_grad()
def features(path):
    import hashlib
    cache=out/'features'/(hashlib.sha256(path.read_bytes()).hexdigest()+'.pt')
    if cache.exists():return torch.load(cache,weights_only=True)
    cache.parent.mkdir(exist_ok=True)
    p=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-threads','2','-'],stdout=subprocess.PIPE)
    colors=[]; block=[]; feats=[]
    for _ in range(20):
        raw=p.stdout.read(24*512*768*3)
        x=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy().reshape(24,512,768,3)).cuda().permute(0,3,1,2)
        colors.append(x.float().mean((2,3)).cpu())
        block.append(torch.nn.functional.avg_pool2d(x.float(),32).cpu())
        vec=model.encode_image(norm(transform(x).float()/255)).float()
        feats.append(torch.nn.functional.normalize(vec,dim=-1).cpu())
    assert not p.stdout.read(1) and p.wait()==0
    value=[torch.cat(a) for a in (colors,block,feats)]
    torch.save(value,cache)
    return value

def score(d,m):
    color,grid,f=m
    ref,g,fd=d
    return dict(first_roll=float(((grid[69]-grid[68])-(g[69]-g[68])).abs().mean()),
                color_self_late=float((color[320:]-color[:21].mean(0)).abs().mean()),
                color_late=float((color-ref).abs()[320:].mean()),
                color_early=float((color-ref).abs()[21:33].mean()),
                seam_transition_error=float(((grid[21]-grid[20])-(g[21]-g[20])).abs().mean()),
                early_grid_error=float((grid-g).abs()[21:33].mean()),
                late_clip_first=float((f[320:]*f[:1]).sum(-1).mean()),
                late_clip_dense=float((f[320:]*fd[320:]).sum(-1).mean()))


import time
start=time.monotonic()
scene=int(sys.argv[1])
teacher=features(out/f'{scene}_11_dense.mp4')
z=torch.load(out/f'{scene}_11_dense.pt',weights_only=True).double()
result={}
indices=list(map(int,sys.argv[2:])) if len(sys.argv)>2 else (11,21)
variants=('original',) if len(sys.argv)>2 else ('original','trained')
import os
if os.environ.get('ABLATION_COMPONENT'): variants=(os.environ['ABLATION_COMPONENT'],)
for idx in indices:
    for variant in variants:
        label=f'{scene}_{idx}_{variant}'
        m=features(out/(label+'.mp4'))
        row=score(teacher,m)
        row['late_grid_error']=(m[1][320:]-teacher[1][320:]).abs().mean().item()
        # Use the same low-motion Dense cells to compare both variants.
        # Rank each spatial cell by its temporal RGB variation across the late interval.
        motion=teacher[1][319:].diff(dim=0).abs().mean((0,1))
        stable_cells=motion.flatten().argsort(stable=True)[:motion.numel()//3]
        err=(m[1][320:]-teacher[1][320:]).abs().mean((0,1))
        row['late_low_motion_grid_error']=err.flatten()[stable_cells].mean().item()
        zz=torch.load(out/(label+'.pt'),weights_only=True).double()
        row['latent_nmse']=((zz-z).square().sum()/z.square().sum()).item()
        result[label]=row
(out/(f'{variants[0]}_quality_{scene}_{indices[0]}.json' if len(sys.argv)>2 else f'quality_{scene}.json')).write_text(json.dumps(dict(rows=result,seconds=time.monotonic()-start),indent=2))
