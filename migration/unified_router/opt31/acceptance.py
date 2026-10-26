"""Read-only results/media acceptance audit; no inference or checkpoint writes."""
import json,math,statistics,subprocess,time
from pathlib import Path
from .setup import ROOT
from .review import REVIEW

def main():
 manifest=json.loads((ROOT/'manifest.json').read_text());rows=[];errors=[];start=time.time()
 for c in manifest['configurations']:
  for case in range(manifest['cases']):
   h,w,s=c['height'],c['width'],c['seconds'];folder=ROOT/f'e2e/{s}s/{h}x{w}/case{case}';d=json.loads((folder/'result.json').read_text());stem=f'{h}x{w}_{s:02d}s_case{case}'
   assert d['completed']and d['frames']==s*16 and d['history_chunks']==3
   for name,v in d['variants'].items():
    assert v['first_chunk_dense_exact']and v['repeats_exact']
    assert len(v['samples'])==(6 if name=='Dense'else 3)and all(math.isfinite(x)and x>0 for x in v['samples'])
    if name!='Dense':
     q=v['quality'];frames=q['per_frame'];assert len(frames)==s*16 and all(x['mse_rgb01']==0 for x in frames[:9]);assert all(math.isfinite(x['mse_rgb01'])and math.isfinite(x['lpips'])for x in frames)
     assert abs(q['sparse_frames_only']['mse_rgb01']-statistics.mean(x['mse_rgb01']for x in frames[9:]))<1e-10
     assert abs(q['sparse_frames_only']['lpips']-statistics.mean(x['lpips']for x in frames[9:]))<1e-10
   media=[]
   for suffix,columns in [('Dense_Q32',2),('Dense_Q64',2),('Dense_Q32_Q64',3)]:
    p=REVIEW/f'{stem}_{suffix}.mp4'
    try:
     stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate,duration','-of','json',str(p)]))['streams'][0]
     assert int(stream['width'])==columns*w and int(stream['height'])==h and int(stream['nb_frames'])==s*16 and stream['r_frame_rate']=='16/1'and abs(float(stream['duration'])-s)<1e-6,stream
     if columns==3:
      subprocess.run(['ffmpeg','-nostdin','-hide_banner','-v','error','-xerror','-threads','1','-i',str(p),'-f','null','-'],check=True,timeout=180,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
     media.append(dict(path=str(p),metadata_correct=True,full_decode_checked=columns==3))
    except Exception as e:errors.append(dict(path=str(p),error=repr(e)))
   rows.append(dict(configuration=c,case=case,prompt_id=d['prompt_id'],metric_recomputed=True,media=media));print('checked',stem,flush=True)
 report=dict(passed=not errors,cases=len(rows),videos=sum(len(r['media'])for r in rows),triple_full_decodes=sum(m['full_decode_checked']for r in rows for m in r['media']),errors=errors,rows=rows,seconds=time.time()-start,scope='Metric consistency, fixed-seed repeat checks recorded by evaluator, all MP4 dimensions/rate/frame-count/duration, full decode of all three-column videos. Does not certify perceptual quality.')
 out=REVIEW/'acceptance';out.mkdir(exist_ok=True);(out/'integrity.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k!='rows'},flush=True)
 if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
