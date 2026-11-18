"""Compare one overlapped shared slot to a complete one-page lookahead in CUDA."""
from pathlib import Path
root=Path(__file__).parent
src=(root.parent/'fixed15_baseline_v1/warp_exact.cu').read_text()
s=src.replace('__shared__ bf kv[2][64*128];','extern __shared__ bf shared[];\n    using Page = bf[2][64*128];\n    Page* buffer=reinterpret_cast<Page*>(shared);')
i=s.index('    stage(kv[0]');j=s.index('        Four score',i)
s=s[:i]+'''    stage(buffer[0][0],K,list[0],h);
    stage(buffer[0][1],V,list[0],h);
    asm volatile("cp.async.commit_group;":::"memory");
    for(int pg=0;pg<count;pg++) {
        Page& kv=buffer[pg%2];
        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        if(pg+1<count){
            int next=list[pg+1];
            stage(buffer[(pg+1)%2][0],K,next,h);
            stage(buffer[(pg+1)%2][1],V,next,h);
        }
        asm volatile("cp.async.commit_group;":::"memory");
'''+s[j:]
i=s.index('        asm volatile("cp.async.wait_group 0;', s.index('A high'))
j=s.index('        #pragma unroll',i)
s=s[:i]+s[j:]
s=s.replace('        __syncthreads();\n    }\n    #pragma','    }\n    #pragma').replace('    exact<<<dim3(72,12),128,0,s>>>','    static const auto optin=cudaFuncSetAttribute(exact,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);\n    if(optin!=cudaSuccess)return optin;\n    exact<<<dim3(72,12),128,65536,s>>>')
(root/'lookahead.cu').write_text(s)
def fragments(shared):
    s=src
    i=s.index('    stage(kv[0]');j=s.index('        Four score',i)
    setup='''    for(int pg=0;pg<count;pg++) {
        int page=list[pg];
        auto fk=reinterpret_cast<const uint4*>(K)+(page*12+h)*1024;
        auto fv=reinterpret_cast<const uint4*>(V)+(page*12+h)*1024;
'''
    if shared:
        setup+='''        for(int m=threadIdx.x;m<1024;m+=128){
            unsigned ak=__cvta_generic_to_shared(reinterpret_cast<uint4*>(kv[0])+m);
            unsigned av=__cvta_generic_to_shared(reinterpret_cast<uint4*>(kv[1])+m);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ak),"l"(fk+m));
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(av),"l"(fv+m));
        }
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
        __syncthreads();
'''
    s=s[:i]+setup+s[j:]
    i=s.index('                unsigned addr=');j=s.index('                mma(score',i)
    ksrc='reinterpret_cast<const uint4*>(kv[0])' if shared else 'fk'
    vsrc='reinterpret_cast<const uint4*>(kv[1])' if shared else 'fv'
    s=s[:i]+f'''                uint4 u={ksrc}[(k*4+n/2)*32+l];
                B b[2]={{{{{{u.x,u.y}}}},{{{{u.z,u.w}}}}}};
'''+s[j:]
    i=s.index('        asm volatile("cp.async.wait_group 0;',s.index('A high'));j=s.index('        #pragma unroll',i)
    s=s[:i]+s[j:]
    i=s.index('                unsigned addr=');j=s.index('                mma(tmp',i)
    s=s[:i]+f'''                uint4 u={vsrc}[(k*8+n/2)*32+l];
                b[k][0]=B{{{{u.x,u.y}}}}; b[k][1]=B{{{{u.z,u.w}}}};
'''+s[j:]
    if not shared:s=s.replace('        __syncthreads();','')
    return s
for name in ('direct','shared'):
    (root/f'{name}.cu').write_text(fragments(name=='shared'))
s=fragments(True).replace('__shared__ bf kv[2][64*128];','extern __shared__ uint4 pool[];\n    using Slot=uint4[2][1024];\n    Slot* slots=reinterpret_cast<Slot*>(pool);')
i=s.index('    for(int pg=0;pg<count;pg++)');j=s.index('        Four score[8]',i)
s=s[:i]+'''    for(int step=-1;step<count;step++){
        int pg=step;
        if(step>=0) {
            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
        }
        if(step+1<count){
            int page=list[step+1];
            int slot=(step+1)&1;
            const uint4* fk=reinterpret_cast<const uint4*>(K)+(page*12+h)*1024;
            const uint4* fv=reinterpret_cast<const uint4*>(V)+(page*12+h)*1024;
            for(int m=threadIdx.x;m<1024;m+=128){
                unsigned ak=__cvta_generic_to_shared(&slots[slot][0][m]);
                unsigned av=__cvta_generic_to_shared(&slots[slot][1][m]);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ak),"l"(fk+m));
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(av),"l"(fv+m));
            }
        }
        asm volatile("cp.async.commit_group;":::"memory");
        if(step<0)continue;
        Slot& kv=slots[step&1];
'''+s[j:]
s=s.replace('        __syncthreads();\n    }\n    #pragma','    }\n    #pragma').replace('    exact<<<dim3(72,12),128,0,s>>>','    static const auto optin=cudaFuncSetAttribute(exact,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);\n    if(optin!=cudaSuccess)return optin;\n    exact<<<dim3(72,12),128,65536,s>>>')
(root/'shared_overlap.cu').write_text(s)
def widen_av(s):
    i=s.index('        for(int n=0;n<16;n+=2)')
    j=s.index('        }\n',s.index('for(int j=0;j<4;j++)out',i))
    return s[:i]+'''        for(int n=0;n<16;n+=4) {
            Four tmp[4]={}; B operands[4][4];
            #pragma unroll
            for(int k=0;k<4;k++){
                #pragma unroll
                for(int t=0;t<4;t+=2){
                    uint4 u=reinterpret_cast<const uint4*>(kv[1])[(k*8+(n+t)/2)*32+l];
                    operands[k][t]=B{{u.x,u.y}}; operands[k][t+1]=B{{u.z,u.w}};
                    mma(tmp[t],high[k],operands[k][t]);
                    mma(tmp[t+1],high[k],operands[k][t+1]);
                }
            }
            #pragma unroll
            for(int k=0;k<4;k++)
                #pragma unroll
                for(int t=0;t<4;t++)mma(tmp[t],low[k],operands[k][t]);
            #pragma unroll
            for(int t=0;t<4;t++)
                #pragma unroll
                for(int j=0;j<4;j++)out[n+t].x[j]+=tmp[t].x[j];
'''+s[j:]
(root/'wide.cu').write_text(widen_av(fragments(True)))
(root/'wide_overlap.cu').write_text(widen_av(s))
