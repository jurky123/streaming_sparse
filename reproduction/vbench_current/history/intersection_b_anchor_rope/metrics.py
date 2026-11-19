import json,sys,subprocess
from pathlib import Path
import torch
import numpy as np
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
    return [torch.cat(a) for a in (colors,block,feats)]

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
i=int(sys.argv[1]); seed=701+i
files={k:root/'videos'/k/'long'/f'{i}_{seed}.mp4' for k in ('dense','lightforcing')}
files['baseline']=out/f'baseline_{i}_{seed}.mp4'
files['rebased']=out/f'rebased_{i}_{seed}.mp4'
d=features(files['dense'])
result={k:score(d,d if k=='dense' else features(v)) for k,v in files.items()}
(out/f'metrics_{i}.json').write_text(json.dumps(result,indent=2))
