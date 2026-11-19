"""Comparable, timestamped video contact sheets; CPU-only parallel extraction."""
import subprocess,io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]/'artifacts/comparison_20261113'
methods=['dense','ours_latest_summary','ours_latest_summary_tail','ours_latest_rope','ours_latest_rope_tail','ours_latest_cached','ours_latest_cached_tail','ours_latest_stable','ours_latest_optimized','ours_latest_fused','ours_latest_fast','ours','dasa','pisa','solattn','svgear','lightforcing','forcingkv'];frames=[61,240,440]
def extract(job):
 case,method,frame=job
 content=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(ROOT/case/method/'video.mp4'),'-vf',f'select=eq(n\\,{frame}),scale=384:256','-frames:v','1','-threads','1','-f','image2pipe','-vcodec','png','-'])
 return job,Image.open(io.BytesIO(content)).copy()
jobs=[(case,method,frame) for case in ['horse_field','traffic_crossing'] for method in methods if (ROOT/case/method/'result.json').exists() for frame in frames]
with ThreadPoolExecutor(max_workers=8) as pool:images=dict(pool.map(extract,jobs))
for case in ['horse_field','traffic_crossing']:
 sheet=Image.new('RGB',(3*384+130,len(methods)*280+50),'white');draw=ImageDraw.Draw(sheet)
 draw.text((10,10),f'{case}: same inputs / 512 x 768 / 480 frames / seed 101',fill='black')
 for col,frame in enumerate(frames):draw.text((130+col*384,30),f'RGB frame {frame} / {frame/16:.2f}s',fill='black')
 for row,method in enumerate(methods):
  y=50+row*280;draw.text((5,y+100),method,fill='black')
  if (case,method,frames[0]) not in images:draw.text((140,y+100),'INCOMPLETE',fill='red');continue
  for col,frame in enumerate(frames):sheet.paste(images[case,method,frame],(130+col*384,y))
 sheet.save(ROOT/f'{case}.contact.png')
