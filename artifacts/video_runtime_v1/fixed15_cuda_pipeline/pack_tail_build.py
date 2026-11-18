from pathlib import Path
root=Path(__file__).parent
s=(root/'transpose_tail.cu').read_text()
i=s.index('    for(int pg=16;');j=s.index('        C score',i)
s=s[:i]+'''    // One slab holds a 64-representative K tile, 16 means and three 16x8 detail tiles.
    constexpr int words=1024+256+48;
    __shared__ uint4 slab[2][words];
    int tiles=(p+15)/16;
    for(int step=0;step<tiles;step++){
        if(step){
            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
        }
        if(step+1<tiles){
            int next=step+1;
            const uint4* input=reinterpret_cast<const uint4*>(KEY)+(h*tiles+next)*words;
            for(int i=threadIdx.x;i<words;i+=128){
                unsigned addr=__cvta_generic_to_shared(slab[next%2]+i);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(input+i));
            }
        }
        asm volatile("cp.async.commit_group;":::"memory");
        if(!step)continue;
        int pg=16*step;
        uint4* data=slab[step%2];
'''+s[j:]
i=s.index('        for(int n=0;n<8;n++){\n            int rep=');j=s.index('        unsigned bits',i)
s=s[:i]+'''        for(int k=0;k<8;k++){
            #pragma unroll
            for(int n=0;n<8;n+=2){
                uint4 v=data[(k*4+n/2)*32+l];
                mma(score[n],q[k],B{{v.x,v.y}});mma(score[n+1],q[k],B{{v.z,v.w}});
            }
        }
'''+s[j:]
s=s.replace('B b=operand(MEAN+(h*hp)*128+n*8+g,128,pg,p,z);','uint4 raw=data[1024+n/2*32+l];\n            B b=(n&1)?B{{raw.z,raw.w}}:B{{raw.x,raw.y}};').replace('B b=operand(DETAIL+(h*hp*3+u)*8+g,24,pg,p,z);','uint2 raw=reinterpret_cast<uint2*>(data+1280)[u*32+l];\n            B b={{raw.x,raw.y}};')
(root/'packed_tail.cu').write_text(s)
