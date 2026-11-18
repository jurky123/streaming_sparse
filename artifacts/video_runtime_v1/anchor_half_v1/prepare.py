"""Launch preparation only for summaries consumed by the specialized first tile and full tiles."""
from pathlib import Path
import subprocess
s=Path('/tmp/jzk_compact_half.cu').read_text()
s=s.replace('int h=id/(tiles*5),tile=id/5%tiles,strip=id%5;', '''// All of tile 0 and K strips 0,1 of tile 1 contain anchor pages.
    int active=tiles*5-7;
    int h=id/active,local=id%active+7,tile=local/5,strip=local%5;''')
s=s.replace('72*48+12*((pages+15)/16)*5','72*48+12*(((pages+15)/16)*5-7)')
out=Path('/tmp/jzk_anchor_prepare.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
