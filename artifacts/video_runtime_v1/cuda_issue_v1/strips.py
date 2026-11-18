"""Pipeline probability conversion while QK advances to the next pair of key fragments."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
src=(root.parent/'cuda_base_layout_v1/fixed15_attention.cu').read_text()
a=src.index('        Four score[8]={};')
b=src.index('        #pragma unroll\n        for(int n=0;n<16;n+=2)',a)
original=src[a:b].replace('        A high[4],low[4];','')
qk=original[original.index('                uint4 u='):original.index('            }\n')]
prob=original[original.index('                float x=exp2ptx'):original.index('            }\n',original.index('                float x=exp2ptx'))]
# First page computes the anchor from the full row; subsequent pages can emit
# probability as soon as a pair of score fragments is ready.
normal='''
            Four scores[2][2]={};
            #pragma unroll
            for(int strip=0;strip<5;strip++) {
                int curr=strip&1, prev=curr^1;
                #pragma unroll
                for(int k=0;k<8;k++) {
                    if(strip<4){
                        uint4 u=kv[0][(k*4+strip)*32+l];
                        if(k==0){scores[curr][0]=Four{};scores[curr][1]=Four{};}
                        mma(scores[curr][0],q[k],B{{u.x,u.y}});
                        mma(scores[curr][1],q[k],B{{u.z,u.w}});
                    }
                    if(strip && k<4){
                        int j=k/2,r=k%2,n=2*(strip-1)+j;
                        float x=exp2ptx(scores[prev][j].x[r*2]*float(1.4426950408889634/11.313708498984761)-anchor[r]);
                        float y=exp2ptx(scores[prev][j].x[r*2+1]*float(1.4426950408889634/11.313708498984761)-anchor[r]);
                        cols[n].x[r*2]+=x;cols[n].x[r*2+1]+=y;
                        high[strip-1].x[2*j+r]=pairbf(x,y);
                        low[strip-1].x[2*j+r]=pairbf(x-__bfloat162float(__float2bfloat16(x)),y-__bfloat162float(__float2bfloat16(y)));
                    }
                }
            }
'''
s=src[:a]+'        A high[4],low[4];\n        if(step==0){\n'+original+'        }else{\n'+normal+'        }\n'+src[b:]
(root/'strips.cu').write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_80','--fmad=false','-lineinfo','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(root/'strips.cu'),'-o','/tmp/jzk_strips.so'],check=True)
