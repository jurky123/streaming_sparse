__global__ void dbg(const float* e,const float* le,const float* mu,const float* code,const float* lt,const bf* basis,float* proj,float* out,float* rho){
    using namespace tailops;
    int h=blockIdx.y,t=blockIdx.x,l=threadIdx.x%32,z=l%4,g=l/4,row=t*64+threadIdx.x/32*16+g;
    tailops::A ah,al;
    float rr[2];
    #pragma unroll
    for(int r=0;r<2;r++){
        int qi=row+8*r;
        rr[r]=divide(1.f,1.f+ex(le[qi*12+h]-lt[h*4608+qi]));rho[qi*12+h]=rr[r];
        #pragma unroll
        for(int k=0;k<2;k++){
            float v=code[(h*4608+qi)*8+z+4*k];
            ah.v[2*k+r]=tf(v);al.v[2*k+r]=tf(v-__uint_as_float(ah.v[2*k+r]));
        }
    }
    #pragma unroll
    for(int n=0;n<16;n++){
        tailops::B b={{tf(__bfloat162float(basis[(h*128+n*8+g)*8+z])),tf(__bfloat162float(basis[(h*128+n*8+g)*8+z+4]))}};
        tailops::C c={};project(c,al,b);project(c,ah,b);
        #pragma unroll
        for(int r=0;r<2;r++)
            #pragma unroll
            for(int j=0;j<2;j++){
                int d=n*8+2*z+j,qi=row+8*r;
                int addr=(qi*12+h)*128+d;
                proj[addr]=c.v[2*r+j];
                out[addr]=e[addr]*(1.f-rr[r])+(mu[(h*4608+qi)*128+d]+c.v[2*r+j])*rr[r];
            }
    }
}
extern "C" void debug(const float* e,const float* le,const float* mu,const float* code,const float* lt,const bf* basis,float* proj,float* out,float* rho,cudaStream_t s){
    dbg<<<dim3(72,12),128,0,s>>>(e,le,mu,code,lt,basis,proj,out,rho);
}
