"""Align the 16-page compensation mask with each lane's eight MMA fragments."""
from pathlib import Path
import sys
import subprocess
root=Path(__file__).parent
src=(root.parent/'phase_boundary_v1/control.cu').read_text()
name=sys.argv[1]
if name=='skip_full':
    src=src.replace('        uint4* data=slab[step%2];','''        unsigned selected=mask[pg/32]>>(pg%32);
        if((selected & 0xffffu)==0xffffu)continue;
        uint4* data=slab[step%2];''')
elif name in ('mean_fma','unit_fade'):
    update='am[n].v[j]=am[n].v[j]*fade[j/2]+c.v[j];'
    if name=='mean_fma':
        src=src.replace(update, 'am[n].v[j]=__fmaf_rn(am[n].v[j],fade[j/2],c.v[j]);')
    else:
        a=src.index('        #pragma unroll\n        for(int n=0;n<16;n++){')
        b=src.index('        #pragma unroll\n        for(int j=0;j<4;j++)ac.v[j]',a)
        body=src[a:b]
        src=src[:a]+('        if(__all_sync(~0u,fade[0]==1.f && fade[1]==1.f)){\n'
            +body.replace(update,'am[n].v[j]+=c.v[j];')+'        }else{\n'+body+'        }\n')+src[b:]
elif name in ('lane_mask','predicate','unmasked_tile','unmasked_half','lane_fma'):
    src=src.replace('if(threadIdx.x<16)mask[threadIdx.x]=0;', '''if(threadIdx.x<16)
        mask[threadIdx.x]=(threadIdx.x==p/32 && p%32)?(~0u<<(p%32)):0u;''')
    src=src.replace('unsigned bits=mask[pg/32];','unsigned bits=mask[pg/32]>>((pg%32)+z/2);')
    src=src.replace('int page=pg+n*2+z/2;', '').replace('page<p && !(bits&(1u<<(page%32)))','!(bits&(1u<<(2*n)))')
    if name in ('unmasked_tile','unmasked_half'):
        src=src.replace('unsigned bits=mask[pg/32]>>((pg%32)+z/2);',
                        'unsigned selected=(mask[pg/32]>>(pg%32))&0xffffu;\n        unsigned bits=selected>>(z/2);')
        # Apply the page mask only in groups that contain selected pages.
        # Finite logits and omitted (-inf) logits both retain the same value after scaling.
        width=8 if name=='unmasked_tile' else 4
        a=src.index('        A hi[4],lo[4];')
        src=src[:a]+f"""        #pragma unroll
        for(int first=0;first<8;first+={width}) {{
            if(selected & ({(1<<(2*width))-1}u<<(2*first))) {{
                #pragma unroll
                for(int n=first;n<first+{width};n++) {{
                    #pragma unroll
                    for(int j=0;j<4;j++)
                        if(bits&(1u<<(2*n)))score[n].v[j]=-CUDART_INF_F;
                }}
            }}
        }}
"""+src[a:]
        src=src.replace('(!(bits&(1u<<(2*n))))?__fmul_rn(score[n].v[r*2+j],0.12751743082459868f):-CUDART_INF_F',
                        '__fmul_rn(score[n].v[r*2+j],0.12751743082459868f)')
    if name=='predicate':
        src=src.replace('float v=(!(bits&(1u<<(2*n))))?__fmul_rn(score[n].v[r*2+j],0.12751743082459868f):-CUDART_INF_F;',
                        '''float v=__fmul_rn(score[n].v[r*2+j],0.12751743082459868f);
                    unsigned omitted=-(unsigned)((bits>>(2*n))&1);
                    v=__uint_as_float((__float_as_uint(v)&~omitted)|(0xff800000u&omitted));''')
elif name!='control':
    raise ValueError(name)
if name=='lane_fma':
    src=src.replace('am[n].v[j]=am[n].v[j]*fade[j/2]+c.v[j];',
                    'am[n].v[j]=__fmaf_rn(am[n].v[j],fade[j/2],c.v[j]);')
out=Path('/tmp/jzk_tailmask_'+name+'.cu')
out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
