"""Reuse the rounded mean/detail coefficient pair for its low BF16 term."""
from pathlib import Path
import argparse
import subprocess

root = Path(__file__).resolve().parent
src = (root / 'control.cu').read_text()
old = '''hi[u].v[2*chunk+r]=pair(a,b);
                    lo[u].v[2*chunk+r]=pair(a-__bfloat162float(__float2bfloat16(a)),b-__bfloat162float(__float2bfloat16(b)));'''
new = '''unsigned rounded=pair(a,b);
                    hi[u].v[2*chunk+r]=rounded;
                    lo[u].v[2*chunk+r]=pair(a-__uint_as_float(rounded<<16),b-__uint_as_float(rounded&0xffff0000u));'''
assert src.count(old) == 1
parser = argparse.ArgumentParser()
parser.add_argument('--combine-mask', action='store_true')
args = parser.parse_args()
src = src.replace(old, new)
name = 'coeffreuse'
if args.combine_mask:
    begin = '    auto compute_tile=[&](int step, auto shape){'
    mask = 'unsigned lane_mask=mask[pg/32]>>((pg%32)+2*z);'
    assert src.count(begin) == src.count(mask) == 1
    src = src.replace(begin, '    unsigned owned=mask[l/2]>>(16*(l%2));\n' + begin)
    src = src.replace(mask, 'unsigned lane_mask=__shfl_sync(~0u,owned,step)>>(2*z);')
    name = 'coeffmask'
path = Path(f'/tmp/jzk_page_{name}.cu')
path.write_text(src)
subprocess.run(['nvcc', '-O3', '--fmad=false', '-std=c++17', '-arch=sm_80', '-shared',
                '-Xcompiler=-fPIC', str(path), '-o', str(path.with_suffix('.so'))], check=True)
