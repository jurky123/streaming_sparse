"""Render only completed local/video results and timestamped full-protocol frames."""
import json,subprocess,io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1]/'artifacts/group_response_178f95a8';raw=Path('/tmp/group_response_results');groups=['G-K','G-VR','G-KVR','G-ORDER'];cases=[f'{c}_s{s}' for c in ['horse_field','traffic_crossing'] for s in [101,303]]
local=json.loads((root/'local_aggregate.json').read_text());lookup={(x['group'],x['panel']):x for x in local}
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
x=np.arange(4)
for j,panel in enumerate(['mass','risk']):axes[0].bar(x+(j-.5)*.35,[lookup[g,panel]['nmse'] for g in groups],width=.35,label=panel)
axes[0].set_title('L1: same-state 10% mixed NMSE');axes[0].legend();axes[0].set_xticks(x,groups)
for j,level in enumerate(['L2','L3']):
 vals=[]
 for g in groups:
  rr=[json.loads((raw/'video'/c/g/level/'result.json').read_text()) for c in cases];vals.append(np.mean([r['quality']['lpips'] for r in rr]))
 axes[j+1].bar(x,vals);axes[j+1].set_xticks(x,groups);axes[j+1].set_title(level+': RGB [21,101) LPIPS vs matched Dense')
for ax in axes:ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('178f95a8 | frozen summary_pretrained | 338 history pages | 512×768 | 9 chunks')
fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
def extract(job):
 case,g,level,frame=job;p=raw/'video'/case/g/level/'video.mp4'
 data=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(p),'-vf',f'select=eq(n\\,{frame}),scale=384:256','-frames:v','1','-threads','1','-f','image2pipe','-vcodec','png','-'])
 return job,Image.open(io.BytesIO(data)).copy()
frames=[21,61,100];jobs=[(c,g,l,f) for c in cases for g in groups for l in ['Dense','L2','L3'] for f in frames]
with ThreadPoolExecutor(max_workers=8) as pool:images=dict(pool.map(extract,jobs))
for c in cases:
 im=Image.new('RGB',(1260,12*280+40),'white');draw=ImageDraw.Draw(im);draw.text((8,8),c+'; columns: RGB21 / 61 / 100',fill='black')
 for i,(g,l) in enumerate((g,l) for g in groups for l in ['Dense','L2','L3']):
  draw.text((4,65+i*280),g+' '+l,fill='black')
  for j,f in enumerate(frames):im.paste(images[c,g,l,f],(108+j*384,40+i*280))
 im.save(root/(c+'.contact.png'))
im=Image.new('RGB',(4*384+100,4*280+40),'white');draw=ImageDraw.Draw(im)
for j,c in enumerate(cases):draw.text((100+j*384,10),c,fill='black')
for i,g in enumerate(groups):
 draw.text((5,65+i*280),g,fill='black')
 for j,c in enumerate(cases):im.paste(images[c,g,'L3',61],(100+j*384,40+i*280))
im.save(root/'L3_frame61.png')
