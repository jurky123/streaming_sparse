"""Direct MP4 review delivery from completed results, with a plain-text index."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .setup import ROOT
from ..opt30.io import atomic_json
REVIEW=Path(__file__).resolve().parents[3]/'artifacts/opt31_video_review'

def index():
 REVIEW.mkdir(parents=True,exist_ok=True)
 lines=['Opt31 视频验收目录', '分辨率命名为 高x宽，16fps。用 VLC / mpv / 系统播放器直接打开 MP4。', '三列顺序：左 Dense，中 Q32 Best，右 Q64 FastReference；各列保持原始分辨率。', 'Best 指本次冻结的 Opt30 q32_full（全29层 attention 验证误差最低）；Q64 为此前最快参照。', '另有两列版本：Dense_Q32、Dense_Q64，可放大观察细节。', '各分辨率/时长同一案例采用相同 prompt、seed；不同长度不保证短视频为长视频前缀。', '新结果完成后自动加入。三列版基于现有 CRF18 两列视频重新编码（CRF18），仅供视觉验收；PSNR/LPIPS始终取未压缩RGB。', '', '已完成评测：']
 for result in sorted((ROOT/'e2e').glob('*/*/*/result.json')):
  d=json.loads(result.read_text())
  if not d.get('completed'):continue
  stem=f'{d["height"]}x{d["width"]}_{d["duration_seconds"]:02d}s_case{d["case"]}'
  lines+=['',f'{stem} | prompt_id={d["prompt_id"]}']
  for old,new in [('Best','Q32'),('FastReference','Q64')]:
   src=result.parent/f'Dense_{old}.mp4';dst=REVIEW/f'{stem}_Dense_{new}.mp4'
   if src.exists()and not dst.exists():os.link(src,dst)
   if dst.exists():lines.append(dst.name)
  triple=REVIEW/f'{stem}_Dense_Q32_Q64.mp4';lines.append(triple.name if triple.exists()else'三列版：整理中')
  for name,label in [('Best','Q32'),('FastReference','Q64')]:
   row=d['variants'][name];q=row['quality']['sparse_frames_only'];lines.append(f'  {label}: PSNR={q["psnr_db"]:.3f} LPIPS={q["lpips"]:.4f} 生成加速={row["speedup_vs_dense"]:.3f}x 含VAE加速={row["speedup_with_vae"]:.3f}x')
 (REVIEW/'视频索引.txt').write_text('\n'.join(lines)+'\n')
 return REVIEW/'视频索引.txt'

def build(result):
 d=json.loads(result.read_text());h,w=d['height'],d['width'];stem=f'{h}x{w}_{d["duration_seconds"]:02d}s_case{d["case"]}'
 out=REVIEW/f'{stem}_Dense_Q32_Q64.mp4'
 if out.exists():return
 a=result.parent/'Dense_Best.mp4';b=result.parent/'Dense_FastReference.mp4';tmp=out.with_suffix('.partial.mp4')
 filters=f'[1:v]crop={w}:{h}:{w}:0[right];[0:v][right]hstack=inputs=2[all];[all]drawbox=x={w}:y=0:w=220:h=35:color=black:t=fill,drawtext=text=Q32_Best:x={w+10}:y=10:fontcolor=white:fontsize=18,drawbox=x={2*w}:y=0:w=240:h=35:color=black:t=fill,drawtext=text=Q64_FastReference:x={2*w+10}:y=10:fontcolor=white:fontsize=18[out]'
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-threads','2','-i',str(a),'-threads','2','-i',str(b),'-filter_complex_threads','1','-filter_complex',filters,'-map','[out]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
 try:
  subprocess.run(cmd,check=True,timeout=900)
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(tmp)]))['streams'][0]
  assert int(info['width'])==3*w and int(info['height'])==h and int(info['nb_frames'])==d['frames'] and info['r_frame_rate']=='16/1',info
  tmp.replace(out);print('video_ready',out,flush=True)
 finally:
  if tmp.exists():tmp.unlink()

def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();REVIEW.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=open(REVIEW/'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 failed={}
 while True:
  index()
  snapshot=set((ROOT/'e2e').glob('*/*/*/result.json'))
  for result in sorted(snapshot,key=lambda p:(int(p.parents[2].name[:-1]),str(p))):
   if str(result)in failed:continue
   try:build(result);index()
   except Exception as e:failed[str(result)]=repr(e);atomic_json(REVIEW/'errors.json',failed);print('review_error',result,repr(e),flush=True)
  q=json.loads((ROOT/'queue.json').read_text())
  if not a.watch:break
  if q['stage']in ['completed','complete_with_failures']:
   # Jobs can finish while earlier videos are encoding. Drain the new results
   # before exiting, even after the evaluation supervisor has completed.
   if set((ROOT/'e2e').glob('*/*/*/result.json'))<=snapshot:break
   continue
  time.sleep(15)
 print(index(),flush=True)
if __name__=='__main__':main()
