"""Diagnostic: feed the mean's hi/lo coefficients to all three detail MMAs."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
source = (root / 'control.cu').read_text()
old = 'mma(c,hi[1+u],b);mma(c,lo[1+u],b);'
assert source.count(old) == 1
source = source.replace(old, 'mma(c,hi[0],b);mma(c,lo[0],b);')
path = Path('/tmp/jzk_page_detailproxy.cu')
path.write_text(source)
subprocess.run(['nvcc', '-O3', '--fmad=false', '-std=c++17', '-arch=sm_80', '-shared',
                '-Xcompiler=-fPIC', str(path), '-o', str(path.with_suffix('.so'))], check=True)
