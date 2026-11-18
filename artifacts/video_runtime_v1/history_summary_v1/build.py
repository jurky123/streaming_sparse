"""Prepare current K/V and only summary tiles touching the 72 current pages."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
src=Path('/tmp/jzk_compact_half.cu').read_text()
s=src.replace('int h=id/(tiles*5),tile=id/5%tiles,strip=id%5;', '''int first=(pages-72)/16,active=tiles-first;
    int h=id/(active*5),tile=first+id/5%active,strip=id%5;''').replace('12*((pages+15)/16)*5','12*(((pages+15)/16)-(pages-72)/16)*5')
out=Path('/tmp/jzk_history_summary.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
