"""Synchronized early 2x2 comparison, source RGB frames21:101, no grading."""
import os,subprocess,json
from pathlib import Path
ROOT=Path('artifacts/high_sparse_train_20261030/comparisons');ROOT.mkdir(exist_ok=True)
def main():
    rows=[];font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    for case in ['basketball_court','butterfly_garden']:
        sources=[(Path('artifacts/compensation_stable_accept_20261030')/case/'stable.mp4','Previous 64.9 percent + compensation')]+[(Path('artifacts/high_sparse_train_20261030/acceptance/early')/case/(mode+'.mp4'),label) for mode,label in [('baseline','90 percent | matched baseline'),('value','90 percent | learned V only'),('joint','90 percent | learned route and V')]]
        dest=ROOT/(case+'_early90.mp4')
        if not dest.exists():
            filters=[];cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-nostdin','-filter_complex_threads','2']
            for i,(path,label) in enumerate(sources):
                cmd+=['-threads','2','-i',str(path)]
                filters.append(f"[{i}:v]trim=start_frame=21:end_frame=101,setpts=PTS-STARTPTS,setsar=1,pad=iw:ih+60:0:60:black,drawtext=fontfile={font}:text='{label.replace('%','percent')}':fontsize=23:fontcolor=white:x=12:y=7,drawtext=fontfile={font}:text='Source frame %{{eif\\:n+21\\:d}} | seed 1001':fontsize=18:fontcolor=white:x=12:y=37[v{i}]")
            filters.append('[v0][v1]hstack=inputs=2[top]');filters.append('[v2][v3]hstack=inputs=2[bottom]');filters.append('[top][bottom]vstack=inputs=2[out]')
            temp=dest.with_suffix('.tmp.mp4');cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
            subprocess.run(cmd,check=True);os.replace(temp,dest)
        check=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-i',str(dest),'-map','0:v:0','-f','null','-','-progress','pipe:1'],capture_output=True,text=True,check=True)
        frames=[int(s.split('=')[1]) for s in check.stdout.splitlines() if s.startswith('frame=')];assert frames[-1]==80
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','2','-i',str(dest),'-frames:v','1','-threads','2',str(ROOT/(case+'.png'))],check=True)
        rows.append(dict(case=case,path=str(dest),frames=80,fps=16,source_frames=[21,101],bytes=dest.stat().st_size,layout='top-left previous64.9comp; top-right90baseline; bottom-left90value; bottom-right90joint',sources=[str(p) for p,_ in sources]));print(rows[-1],flush=True)
    (ROOT/'manifest.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
