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

template<bool Rotate, bool Pack = false>
__global__ void rms_row_kernel(const uint2* x, const uint2* weight,
                               uint2* out, int rows, float epsilon, const double2* freq, int gh, int gw, int start, uint4* packed = nullptr) {
  __shared__ __align__(16) unsigned short stage[Pack ? 12 : 1][Pack ? 16 : 1][Pack ? 128 : 1];
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
    if constexpr (Pack) {
      int col = (4*lane) ^ ((threadIdx.y%8)*8);
      *reinterpret_cast<uint2*>(&stage[i][threadIdx.y][col]) = result.packed;
    }
  }
  if constexpr (Pack) {
    __syncthreads();
    // Each warp emits one contiguous 16-column K fragment at a time.
    int t = threadIdx.y*32 + lane;
    for (int i=t; i<12*256; i+=512) {
      int head=i/256, tile=(i%256)/32, quad=lane/8;
      int r=(quad/2)*8+lane%8, col=tile*16+(quad%2)*8;
      unsigned addr=__cvta_generic_to_shared(&stage[head][r][col^((r%8)*8)]);
      uint4 a;
      asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
          :"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(addr));
      packed[(blockIdx.x/4*12+head)*1024+tile*128+(blockIdx.x%4)*32+lane]=a;
    }
  }
}

extern "C" void rms_row(const void* x, const void* weight, void* out,
                        int rows, float epsilon, cudaStream_t stream) {
  rms_row_kernel<false><<<(rows+3)/4, dim3(32,4), 0, stream>>>(
      (const uint2*)x, (const uint2*)weight, (uint2*)out, rows, epsilon, nullptr, 0, 0, 0);
}

extern "C" void rms_rope(const void* x,const void* weight,void* out,int rows,float epsilon,
                         const double2* freq,int gh,int gw,int start,cudaStream_t stream) {
  rms_row_kernel<true><<<(rows+3)/4,dim3(32,4),0,stream>>>(
      (const uint2*)x,(const uint2*)weight,(uint2*)out,rows,epsilon,freq,gh,gw,start);
}

extern "C" void rms_rope_pack(const void* x,const void* weight,void* out,int rows,float epsilon,
                              const double2* freq,int gh,int gw,int start,uint4* packed,cudaStream_t stream) {
  rms_row_kernel<true,true><<<rows/16,dim3(32,16),0,stream>>>(
      (const uint2*)x,(const uint2*)weight,(uint2*)out,rows,epsilon,freq,gh,gw,start,packed);
}
