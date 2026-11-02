"""Frame-aligned labeled comparison; CPU only, no changes to colors or FPS."""
import json,subprocess,os
from pathlib import Path
ROOT=Path('artifacts/compensation_comparison_20261030');ROOT.mkdir(exist_ok=True)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def main():
    manifest=[]
    for case in ['basketball_court','butterfly_garden']:
        old=Path('artifacts/compensation_probe_20261030/acceptance')/(case+'_s1001')
        inputs=[old/'dense_host.mp4',old/'mass_var.mp4',Path('artifacts/compensation_stable_accept_20261030')/case/'stable.mp4']
        for kind,start,end in [('early',21,101),('full',0,480)]:
            dest=ROOT/f'{case}_{kind}.mp4'
            if not dest.exists():
                filters=[]
                for i,label in enumerate(['Dense reference','Sparse baseline','Sparse + compensation (optimized)']):
                    filters.append(f"[{i}:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,setsar=1,pad=iw:ih+64:0:64:color=black,drawtext=fontfile={FONT}:text='{label}':fontsize=25:fontcolor=white:x=14:y=8,drawtext=fontfile={FONT}:text='Source frame %{{eif\\:n+{start}\\:d}} | seed 1001':fontsize=19:fontcolor=white:x=14:y=39[v{i}]")
                filters.append('[v0][v1][v2]hstack=inputs=3[out]')
                cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-filter_complex_threads','2']
                for p in inputs:cmd+=['-threads','2','-i',str(p)]
                temp=dest.with_suffix('.tmp.mp4')
                cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
                subprocess.run(cmd,check=True);os.replace(temp,dest)
            # Validate the complete encoded stream, count decoded frames.
            r=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-i',str(dest),'-map','0:v:0','-f','null','-','-progress','pipe:1'],capture_output=True,text=True,check=True)
            frames=[int(x.split('=')[1]) for x in r.stdout.splitlines() if x.startswith('frame=')];assert frames[-1]==end-start,(dest,frames)
            manifest.append(dict(case=case,kind=kind,path=str(dest),source_paths=list(map(str,inputs)),source_frames=[start,end],frames=frames[-1],fps=16,duration_s=(end-start)/16,bytes=dest.stat().st_size))
            print(json.dumps(manifest[-1]),flush=True)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Static previews are for navigation, not quality metrics.
    for case in ['basketball_court','butterfly_garden']:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','2','-i',str(ROOT/f'{case}_early.mp4'),'-frames:v','1','-threads','2',str(ROOT/f'{case}_preview.png')],check=True)
if __name__=='__main__':main()
