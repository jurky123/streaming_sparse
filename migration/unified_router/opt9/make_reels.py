import json,subprocess
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024').resolve();out=root/'optimization9';pids=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids']
for variant in ['Packed','Skip2','Skip0','Skip2Fast','Skip0Fast']:
 listing=out/f'concat_{variant}.txt'
 listing.write_text(''.join("file '"+str(out/f'quality/{pid}.seed0/Dense_Opt8_{variant}.mp4')+"'\n"for pid in pids))
 dest=out/f'ALL8_Dense_Opt8Full_Opt9_{variant}.mp4'
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-vf',f"drawtext=text='Opt9 {variant}':x=1672:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.9:boxborderw=5",'-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-an',str(dest)],check=True)
 print(dest.name,flush=True)
