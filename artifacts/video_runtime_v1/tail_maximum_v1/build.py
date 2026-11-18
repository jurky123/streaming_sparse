"""Apply the positive logit scale after reducing the block maximum."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).parent
name=sys.argv[1]
s=(root/'control.cu').read_text()
scale='0.12751743082459868f'
if name in ('raw_max','center_fma'):
    s=s.replace('float nm=mx[r];','float nm=-CUDART_INF_F;')
    s=s.replace(f'__fmul_rn(score[n].v[r*2+j],{scale})','score[n].v[r*2+j]')
    s=s.replace('            fade[r]=',f'            nm=fmaxf(mx[r],__fmul_rn(nm,{scale}));\n            fade[r]=')
    for offset in ('2*r','2*r+1'):
        x=f'score[n].v[{offset}]'
        centered=(f'__fmaf_rn({x},{scale},-nm)' if name=='center_fma' else f'__fmul_rn({x},{scale})-nm')
        s=s.replace('ex('+x+'-nm)','ex('+centered+')')
elif name in ('coeff_mass','direct_mass','direct_order','direct_shared_den','direct_shuffle_den'):
    if name in ('coeff_mass','direct_mass'):
        s=s.replace('partial+=a;partial+=b;','')
        a=s.index('            partial+=__shfl_xor_sync')
        b=s.index('            #pragma unroll',a)
        denominator=s[a:b]
        s=s[:a]+s[b:]
        i=s.index('                #pragma unroll\n                for(int u=0;u<4;u++)')
        s=s[:i]+'                partial+=left[0];partial+=right[0];\n'+s[i:]
        i=s.index('        }\n        #pragma unroll\n        for(int n=0;n<16;n++){')
        s=s[:i]+denominator+s[i:]
    if name!='coeff_mass':
        # Reorder representative keys at the producer so one lane owns the four weights of a page.
        s=s.replace('int token=tile*64+strip*16+row;', """int pos=strip*16+row;
            int page_in_tile=8*(pos/32)+2*((pos%8)/2)+(pos%32)/16;
            int token=tile*64+4*page_in_tile+2*((pos/8)%2)+pos%2;""")
        s=s.replace('((pg%32)+z/2)','((pg%32)+2*z)').replace('(1u<<(2*n))','(1u<<(8*(n/4)+(n%4)/2))')
        a=s.index('                #pragma unroll\n                for(int bit=1;bit<=2;bit*=2){')
        b=s.index('                float sums[4]',a)
        s=s[:a]+s[b:]
    if name in ('direct_shared_den','direct_shuffle_den'):
        s=s.replace('partial+=a;partial+=b;','')
        target=s.index('            partial+=__shfl_xor_sync')
        if name=='direct_shared_den':
            s=s.replace('    A q[8];C am', '    __shared__ float2 transpose[4][8][4][4];\n    A q[8];C am')
            target=s.index('            partial+=__shfl_xor_sync')
            exchange="""            #pragma unroll
            for(int half=0;half<2;half++) {
                #pragma unroll
                for(int n=0;n<4;n++)transpose[w][g][n][z]=make_float2(score[4*half+n].v[2*r],score[4*half+n].v[2*r+1]);
                __syncwarp();
                #pragma unroll
                for(int n=0;n<4;n++){float2 w=transpose[threadIdx.x/32][g][z][n];partial+=w.x;partial+=w.y;}
                __syncwarp();
            }
"""
        else:
            initial=(root/'control.cu').read_text()
            i=initial.index('                float a[4],b[4];')
            j=initial.index('                float sums[4]',i)
            exchange=('            #pragma unroll\n            for(int chunk=0;chunk<2;chunk++){\n'+initial[i:j]+
                      '                #pragma unroll\n                for(int n=0;n<4;n++){partial+=a[n];partial+=b[n];}\n            }\n')
        s=s[:target]+exchange+s[target:]
else:
    assert name=='control'
source=Path('/tmp/jzk_tailmax_'+name+'.cu')
source.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(source),'-o',str(source.with_suffix('.so'))],check=True)
