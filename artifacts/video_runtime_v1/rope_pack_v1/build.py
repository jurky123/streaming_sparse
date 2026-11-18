"""Produce packed K alongside the native RoPE output; leave V packing in prepare."""
from pathlib import Path
import subprocess
root=Path('/tmp/jzk_history_summary_deployed_base/source/csrc')
def compile(src,name):
    out=Path('/tmp/jzk_ropepack_'+name+'.cu')
    out.write_text(src)
    subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
s=(root/'rms_mean.cu').read_text()
a=s.index('template<bool Rotate>')
b=s.index('extern "C" void rms_row',a)
body=s[a:b].replace('int gw, int start) {','int gw, int start,unsigned* packed) {')
key='    out[row * 384 + i * 32 + lane] = result.packed;'
body=body.replace(key,key+'''
    int tile=lane/4,strip=row/16%4;
    int dest_lane=4*(row%8)+2*(lane%2);
    int vec=((row/64*12+i)*1024+tile*128+strip*32+dest_lane)*4;
    int word=2*(row/8%2)+lane/2%2;
    packed[vec+word]=result.packed.x;
    packed[vec+4+word]=result.packed.y;''')
s=s[:a]+body+s[s.index('extern "C" void rms_rope',b):]
s=s.replace('int start,cudaStream_t stream)', 'int start,unsigned* packed,cudaStream_t stream)').replace('epsilon,freq,gh,gw,start);','epsilon,freq,gh,gw,start,packed);')
compile(s,'direct')
s=(root/'fixed15_attention.cu').read_text()
a=s.index('__device__ __forceinline__ void copy_frag')
b=s.index('template<bool ReuseHistory>',a)
part=s[a:b]
part='\n'.join(l for l in part.split('\n') if not any(k in l for k in ('*>(&keys[','unsigned ka=','asm("ldmatrix.sync.aligned.m8n8.x4.shared','pk[base+')))
compile(s[:a]+part+s[b:],'vonly')
