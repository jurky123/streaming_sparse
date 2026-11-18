// Reduction order follows PyTorch 2.6 Reduce.cuh / SharedReduceOps.h.
// See PYTORCH_LICENSE.txt in this directory. Fixed contiguous D=1536.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void rms_mean_kernel(const __nv_bfloat16* x,float* out,int rows) {
  int lane=threadIdx.x,row=blockIdx.x*blockDim.y+threadIdx.y;
  if(row>=rows)return;
  float s0=0.f,s1=0.f,s2=0.f,s3=0.f;
  // Same four independent accumulators and input-vector stride as PyTorch.
  for(int v=lane;v<384;v+=32){
    const __nv_bfloat16* p=x+row*1536+v*4;
    float a=__bfloat162float(p[0]),b=__bfloat162float(p[1]);
    float c=__bfloat162float(p[2]),d=__bfloat162float(p[3]);
    s0=__fadd_rn(s0,__fmul_rn(a,a));s1=__fadd_rn(s1,__fmul_rn(b,b));
    s2=__fadd_rn(s2,__fmul_rn(c,c));s3=__fadd_rn(s3,__fmul_rn(d,d));
  }
  float s=__fadd_rn(__fadd_rn(__fadd_rn(s0,s1),s2),s3);
  // PyTorch uses increasing shuffle offsets, not the common decreasing tree.
  for(int delta=1;delta<32;delta*=2)s=__fadd_rn(s,__shfl_down_sync(0xffffffff,s,delta));
  if(lane==0)out[row]=__fmul_rn(s,1.f/1536.f);
}
extern "C" void rms_mean(const void* x,void* mean,int rows,cudaStream_t stream){
  rms_mean_kernel<<<(rows+15)/16,dim3(32,16),0,stream>>>((const __nv_bfloat16*)x,(float*)mean,rows);
}

// Each lane keeps its 12 four-element vectors across the original mean tree.
union FourBF16 {
  uint2 packed;
  unsigned short bits[4];
};

template<bool Rotate>
__global__ void rms_row_kernel(const uint2* x, const uint2* weight,
                               uint2* out, int rows, float epsilon, const double2* freq, int gh, int gw, int start, unsigned* pk, unsigned* pv, const uint2* v) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  FourBF16 vectors[12];
  float sums[4] = {};
  #pragma unroll
  for (int i = 0; i < 12; ++i) {
    vectors[i].packed = x[row * 384 + i * 32 + lane];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float v = __bfloat162float(__ushort_as_bfloat16(vectors[i].bits[j]));
      sums[j] = __fadd_rn(sums[j], __fmul_rn(v, v));
    }
  }
  float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  for (int offset = 1; offset < 32; offset *= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) * (1.f / 1536.f) + epsilon);
  double2 coeff[2];
  if (Rotate) {
    #pragma unroll
    for (int j=0; j<2; ++j) {
      int pair=2*lane+j;
      int pos=pair<22 ? row/(gh*gw)+start : pair<43 ? (row/gw)%gh : row%gw;
      coeff[j]=freq[pos*64+pair];
    }
  }
  #pragma unroll
  for (int i = 0; i < 12; ++i) {
    FourBF16 w, result;
    w.packed = weight[i * 32 + lane];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float v = __bfloat162float(__ushort_as_bfloat16(vectors[i].bits[j]));
      float normalized = __bfloat162float(__float2bfloat16_rn(v * inv));
      float scale = __bfloat162float(__ushort_as_bfloat16(w.bits[j]));
      result.bits[j] = __bfloat16_as_ushort(__float2bfloat16_rn(normalized * scale));
    }
    if (Rotate) {
      #pragma unroll
      for (int j=0; j<2; ++j) {
        double re=__bfloat162float(__ushort_as_bfloat16(result.bits[2*j]));
        double im=__bfloat162float(__ushort_as_bfloat16(result.bits[2*j+1]));
        float a=__double2float_rn(re*coeff[j].x-im*coeff[j].y);
        float b=__double2float_rn(re*coeff[j].y+im*coeff[j].x);
        result.bits[2*j]=__bfloat16_as_ushort(__float2bfloat16_rn(a));
        result.bits[2*j+1]=__bfloat16_as_ushort(__float2bfloat16_rn(b));
      }
    }
    out[row * 384 + i * 32 + lane] = result.packed;
    int tok = row % 64, dim = lane * 4;
    int base = (row / 64 * 12 + i) * 8192;
    uint2 val = v[row * 384 + i * 32 + lane];
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      int d = dim + pair*2;
      int key = (d/16*1024 + tok/16*256 + (tok%8*4+d%8/2)*8
                 + tok%16/8*4 + d%16/8*2);
      pk[(base+key)/2] = pair ? result.packed.y : result.packed.x;
    }
    #pragma unroll
    for (int j=0; j<4; ++j) {
      int d=dim+j;
      int value = (tok/16*2048 + d/16*256 + (d%8*4+tok%8/2)*8
                   + d%16/8*4 + tok%16/8*2 + tok%2);
      reinterpret_cast<unsigned short*>(pv)[base+value] =
          static_cast<unsigned short>((j<2 ? val.x : val.y) >> ((j%2)*16));
    }
  }
}

extern "C" void emit(const void* x,const void* weight,void* out,int rows,float epsilon,
           const double2* freq,int gh,int gw,int start,unsigned* pk,unsigned* pv,const uint2* v,cudaStream_t stream) {
  rms_row_kernel<true><<<(rows+3)/4,dim3(32,4),0,stream>>>(
      (const uint2*)x,(const uint2*)weight,(uint2*)out,rows,epsilon,freq,gh,gw,start,pk,pv,v);
}
