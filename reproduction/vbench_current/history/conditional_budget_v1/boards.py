from pathlib import Path
import sys,subprocess
from PIL import Image,ImageDraw
from visual_review import extract
d=Path(__file__).parent;old=d.parent/'rebased_retrain_v1'
for sc in (0,1):
 for t,i in ((10,11),(15,21)):
  files=[old/f'{sc}_{i}_original.mp4',d/f'{sc}_{t}_original.mp4',d/f'{sc}_{t}_trained.mp4',d/f'{sc}_{t}_refit.mp4']
  files=files[:int(sys.argv[1])]
  for label,frames in [('sequence',[0,20,69,128,240,336,416,479]),('boundary',[19,20,21,22,67,68,69,70])]:
   im=Image.new('RGB',(248*8,166*len(files)+20));draw=ImageDraw.Draw(im)
   for j,n in enumerate(frames):draw.text((j*248,2),f'{n/16:.2f}s',fill='white')
   for row,f in enumerate(files):
    for col,x in enumerate(extract(f,frames)):im.paste(x,(col*248,row*166+20))
   im.save(d/f'{sc}_{t}_{label}_{len(files)}.jpg',quality=92)
  if len(files)==4:
   cmd=['ffmpeg','-y','-v','error','-filter_complex_threads','2']
   for f in files:cmd+=['-i',str(f)]
   filt=';'.join(f"[{j}:v]scale=576:384,drawtext=text='{name}':x=6:y=6:fontsize=17:fontcolor=white:box=1:boxcolor=black[v{j}]" for j,name in enumerate(['Fixed original','Table original','Table prior tuning','Table refit']))
   cmd+=['-filter_complex',filt+';[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18',str(d/f'compare_{sc}_{t}.mp4')]
   subprocess.run(cmd,check=True)
