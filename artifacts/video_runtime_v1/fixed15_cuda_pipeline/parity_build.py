from pathlib import Path
root=Path(__file__).parent
s=(root/'fused.cu').read_text()
s=s.replace('partial+=a+b;', 'partial+=a;partial+=b;').replace('partial+=__shfl_xor_sync(~0u,partial,1);partial+=__shfl_xor_sync(~0u,partial,2);','partial+=__shfl_xor_sync(~0u,partial,2);partial+=__shfl_xor_sync(~0u,partial,1);')
s=s.replace('sums[0]+sums[1],sums[0]-sums[1],diffs[0]+diffs[1],diffs[0]-diffs[1]', '(sums[0]+a[1])+b[1],(sums[0]-a[1])-b[1],(diffs[0]+a[1])-b[1],(diffs[0]-a[1])+b[1]')
s=s.replace('sums[2]+sums[3],sums[2]-sums[3],diffs[2]+diffs[3],diffs[2]-diffs[3]', '(sums[2]+a[3])+b[3],(sums[2]-a[3])-b[3],(diffs[2]+a[3])-b[3],(diffs[2]-a[3])+b[3]')
i=s.index('__device__ __forceinline__ void tail')
s=s[:i]+'''__device__ __forceinline__ float divide(float a,float b){
    float c;asm("div.full.f32 %0,%1,%2;":"=f"(c):"f"(a),"f"(b));return c;
}
__device__ __forceinline__ unsigned tf(float x){
    unsigned u;asm("cvt.rna.tf32.f32 %0,%1;":"=r"(u):"f"(x));return u;
}
__device__ __forceinline__ void project(C& c,A a,B b){
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c.v[0]),"+f"(c.v[1]),"+f"(c.v[2]),"+f"(c.v[3])
        :"r"(a.v[0]),"r"(a.v[1]),"r"(a.v[2]),"r"(a.v[3]),"r"(b.v[0]),"r"(b.v[1]));
}
'''+s[i:]
i=s.index('    #pragma unroll\n    for(int r=0;r<2;r++){\n        float safe=fmaxf(den[r],1e-30f);')
j=s.index('\n}\n',i)
s=s[:i]+'''    A high,low;
    float inverse[2],rho[2];
    #pragma unroll
    for(int r=0;r<2;r++){
        float safe=fmaxf(den[r],1e-30f);
        inverse[r]=safe;
        float lt=den[r]>0?mx[r]+log2f(safe)+4.f:-CUDART_INF_F;
        int local=(row+8*r)%64;
        rho[r]=divide(1.f,1.f+ex(le[local]-lt));
        #pragma unroll
        for(int k=0;k<2;k++){
            int src=l-z+z/2+2*k;
            float a=__shfl_sync(~0u,divide(ac.v[2*r],safe),src);
            float b=__shfl_sync(~0u,divide(ac.v[2*r+1],safe),src);
            float v=(z&1)?b:a;
            high.v[2*k+r]=tf(v);
            low.v[2*k+r]=tf(v-__uint_as_float(high.v[2*k+r]));
        }
    }
    #pragma unroll
    for(int n=0;n<16;n++){
        B b={{tf(__bfloat162float(basis[(h*128+n*8+g)*8+z])),
              tf(__bfloat162float(basis[(h*128+n*8+g)*8+z+4]))}};
        C proj={};project(proj,low,b);project(proj,high,b);
        #pragma unroll
        for(int r=0;r<2;r++)
            #pragma unroll
            for(int j=0;j<2;j++){
                int dim=n*8+2*z+j, local=(row+8*r)%64;
                float tv=divide(am[n].v[2*r+j],inverse[r])+proj.v[2*r+j];
                float e=exact[local*128+dim];
                out[((row+8*r)*12+h)*128+dim]=__float2bfloat16(e*(1.f-rho[r])+tv*rho[r]);
            }
    }
'''+s[j:]
(root/'parity.cu').write_text(s)
f=(root/'fused.py').read_text().replace('fixed15_fused.so','fixed15_parity.so').replace("f'fused_{p}.json'","f'parity_{p}.json'").replace("contracts['fallback_tiles']=int(flags.sum())","contracts.update(fallback_tiles=int(flags.sum()),changed=int((old!=new).sum()))")
(root/'parity.py').write_text(f)
