"""Local panels, five-layer L2 heatmap and matched timestamped L3 comparisons."""
import json,csv,io,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parents[1]/'artifacts/group_hierarchy_178f95a8';ROOT=Path('/tmp/group_hierarchy_results')
groups=['G-ORDER','G-K','G-K3-V2-Kinner','G-K-joint','G-K3-V2','G-KVR'];labels=['ORDER','K / K inner','K3V2 / K inner','K / joint inner','K3V2 / joint inner','KVR / joint inner'];colors=['#64748b','#2563eb','#0d9488','#2563eb','#0d9488','#a855f7']
data=json.loads((OUT/'local_aggregate.json').read_text());ix={(r['group'],r['panel']):r for r in data};fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for j,panel in enumerate(['representation','mass','risk']):
 for i,(metric,title) in enumerate([('nmse','Energy-weighted NMSE'),('page_log_denominator_mae','Page log denominator MAE')]):
  ax=axs[i,j];vals=[ix[g,panel][metric] for g in groups];ax.barh(np.arange(6),vals,color=colors);ax.set_yticks(np.arange(6),labels if j==0 else ['']*6);ax.invert_yaxis();ax.set_title(panel+' | '+title);ax.set_xlim(0,max(vals)*1.2);ax.grid(axis='x',alpha=.2)
  for y,v in enumerate(vals):ax.text(v,y,f' {v:.4f}',va='center')
fig.suptitle('120 restored SAME-QKV events | frozen 178f95a8 summary_pretrained | 338 history pages')
fig.savefig(OUT/'local_panels.png',dpi=180);plt.close(fig)
video=list(csv.DictReader((OUT/'video_by_layer.csv').open()));vi={(r['group'],r['level']):r for r in video};layers=[1,8,15,22,29];arr=np.array([[float(vi[g,'L2_'+str(l)]['lpips']) for l in layers] for g in groups]);fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);im=ax.imshow(arr,cmap='YlGnBu');ax.set_xticks(range(5),['layer '+str(l) for l in layers]);ax.set_yticks(range(6),labels)
for i in range(6):
 for j in range(5):ax.text(j,i,f'{arr[i,j]:.4f}',ha='center',va='center',color='white' if arr[i,j]>(arr.max()+arr.min())/2 else 'black')
ax.set_title('L2: 4-case mean LPIPS vs each matched Dense | RGB [21,101)');fig.colorbar(im,ax=ax);fig.savefig(OUT/'L2_layers.png',dpi=180);plt.close(fig)
cases=[f'{c}_s{s}' for c in ['horse_field','traffic_crossing'] for s in [101,303]]
def extract(job):
 c,g,frame=job;p=ROOT/'video'/c/g/'L3/video.mp4';b=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(p),'-vf',f'select=eq(n\\,{frame}),scale=384:256','-frames:v','1','-threads','1','-f','image2pipe','-vcodec','png','-']);return job,Image.open(io.BytesIO(b)).copy()
jobs=[(c,g,f) for c in cases for g in groups for f in [21,61,100]]
with ThreadPoolExecutor(max_workers=8) as pool:images=dict(pool.map(extract,jobs))
for c in cases:
 im=Image.new('RGB',(3*384+180,6*280+40),'white');draw=ImageDraw.Draw(im);draw.text((5,8),c+' L3: RGB21 / 61 / 100',fill='black')
 for i,(g,label) in enumerate(zip(groups,labels)):
  draw.text((4,80+i*280),label,fill='black')
  for j,f in enumerate([21,61,100]):im.paste(images[c,g,f],(180+j*384,40+i*280))
 im.save(OUT/(c+'.L3.contact.png'))
im=Image.new('RGB',(4*384+180,6*280+40),'white');draw=ImageDraw.Draw(im)
for j,c in enumerate(cases):draw.text((180+j*384,10),c,fill='black')
for i,(g,label) in enumerate(zip(groups,labels)):
 draw.text((4,80+i*280),label,fill='black')
 for j,c in enumerate(cases):im.paste(images[c,g,61],(180+j*384,40+i*280))
im.save(OUT/'L3_frame61.png')
