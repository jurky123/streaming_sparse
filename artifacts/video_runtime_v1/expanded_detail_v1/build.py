"""Expand the three signed rank-8 components into four rank-8 representative values."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
s=(root.parent/'page_slope_v1/control.cu').read_text()
s=s.replace('1024+256+48','1024+256+128').replace('2656','2816').replace('*1328','*1408')
s=s.replace('A hi[4],lo[4];','A hi[1],lo[1],ph[4],pl[4];')
point='score[n].v[2*r]=a;score[n].v[2*r+1]=b;'
s=s.replace(point,point+'''
                ph[n/2].v[2*(n%2)+r]=pair(a,b);
                pl[n/2].v[2*(n%2)+r]=pair(a-__bfloat162float(__float2bfloat16(a)),b-__bfloat162float(__float2bfloat16(b)));''')
s=s.replace('for(int u=0;u<4;u++){','for(int u=0;u<1;u++){')
a=s.index('        for(int u=0;u<3;u++){')
b=s.index('\n    }',a)
s=s[:a]+'''        for(int k=0;k<4;k++){
            uint2 high=reinterpret_cast<uint2*>(data+1280)[32*k+l];
            uint2 low=reinterpret_cast<uint2*>(data+1344)[32*k+l];
            B vh={{high.x,high.y}},vl={{low.x,low.y}};
            C c={};
            mma(c,ph[k],vh);mma(c,pl[k],vh);
            mma(c,ph[k],vl);mma(c,pl[k],vl);
            #pragma unroll
            for(int j=0;j<4;j++)ac.v[j]+=c.v[j];
        }'''+s[b:]
a=s.index('    if(strip==4 && t<96)')
b=s.index('\n}',a)
s=s[:a]+'''    if(strip==4){
        for(int i=t;i<512;i+=128){
            int lane=(i/4)%32;
            int pos=16*(i/128)+2*(lane%4)+8*((i/2)%2)+i%2;
            int page=tile*16+8*(pos/32)+2*((pos%8)/2)+(pos%32)/16;
            int rep=2*((pos/8)%2)+pos%2;
            float value=0.f;
            if(page<pages){
                const halfbits* src=detail+(h*hp+page)*24+lane/4;
                float a=__uint_as_float(unsigned(src[0])<<16);
                float b=__uint_as_float(unsigned(src[8])<<16);
                float c=__uint_as_float(unsigned(src[16])<<16);
                value=((rep<2?a:-a)+(rep%2?-b:b))+((rep==0 || rep==3)?c:-c);
            }
            auto high=__float2bfloat16(value);
            reinterpret_cast<__nv_bfloat16*>(dst+1280)[i]=high;
            reinterpret_cast<__nv_bfloat16*>(dst+1344)[i]=__float2bfloat16(value-__bfloat162float(high));
        }
    }'''+s[b:]
out=Path('/tmp/jzk_expanded_detail.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
