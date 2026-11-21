import subprocess,sys
from pathlib import Path
d=Path(__file__).resolve().parent
names=['dense_before','base_before','intersection_fast_before','lightforcing_before','dense_before','base_after','intersection_fast_after','lightforcing_after']
labels=['Dense','Base','Intersection fast','LightForcing']
args=['ffmpeg','-v','error','-y','-filter_complex_threads','2']
filters=[]
for i,name in enumerate(names):
    args+=['-i',str(d/(name+('_5s' if '--short' in sys.argv else '')+'.mp4'))]
    label=labels[i%4]+(' | before' if i<4 else ' | after')
    filters.append(f"[{i}:v]scale=384:256,drawtext=text='{label}':fontsize=17:fontcolor=white:box=1:boxcolor=black@0.7:x=6:y=6[v{i}]")
for j in range(2):
    filters.append(''.join(f'[v{i}]' for i in range(4*j,4*j+4))+f'hstack=4[r{j}]')
filters.append('[r0][r1]vstack=2[out]')
subprocess.run(args+['-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-threads','4','-crf','18',str(d/('comparison_5s.mp4' if '--short' in sys.argv else 'comparison.mp4'))],check=True)
