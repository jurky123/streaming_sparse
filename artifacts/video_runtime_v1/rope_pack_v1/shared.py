"""Gather 16 RoPE rows in shared memory and coalesce packed K through ldmatrix."""
from pathlib import Path
import subprocess
s=Path('/tmp/jzk_ropepack_direct.cu').read_text()
a=s.index('    int tile=lane/4')
b=s.index('\n  }\n}',a)
s=s[:a]+'''    int col=(4*lane)^((threadIdx.y%8)*8);
    *reinterpret_cast<uint2*>(&stage[i][threadIdx.y][col])=result.packed;
  }
  __syncthreads();
  int t=threadIdx.y*32+lane;
  #pragma unroll
  for(int ii=t;ii<12*256;ii+=512){
    int head=ii/256,j=ii%256,quad=lane/8;
    int row_in_tile=(quad/2)*8+lane%8;
    int col=(j/32)*16+(quad%2)*8;
    unsigned ptr=__cvta_generic_to_shared(&stage[head][row_in_tile][col^((row_in_tile%8)*8)]);
    uint4 a;
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(ptr));
    reinterpret_cast<uint4*>(packed)[(blockIdx.x/4*12+head)*1024+j/32*128+(blockIdx.x%4)*32+lane]=a;'''+s[b:]
s=s.replace('  const int lane = threadIdx.x;', '  __shared__ __align__(16) unsigned short stage[12][16][128];\n  const int lane = threadIdx.x;')
s=s.replace('<<<(rows+3)/4,dim3(32,4)','<<<rows/16,dim3(32,16)')
out=Path('/tmp/jzk_ropepack_shared.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
