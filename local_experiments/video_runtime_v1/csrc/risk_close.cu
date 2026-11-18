#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

// All four Risk terms and their weights are nonnegative. Bitwise ordering of
// nonnegative FP32 values gives the same near-tie test as sorting the scores.
template<int Items>
__global__ void close_rows(const float* risk, const long long* own, int* rows,
                           int* count, int pages, int budget) {
    constexpr int threads=128;
    using Sort=cub::BlockRadixSort<unsigned,threads,Items>;
    __shared__ typename Sort::TempStorage scratch;
    __shared__ float top[45];
    const int row=blockIdx.x;
    unsigned key[Items];
    #pragma unroll
    for(int j=0;j<Items;++j) {
        const int i=threadIdx.x*Items+j;
        key[j]=i==own[row] ? 0x7f800000u : i<pages ? __float_as_uint(risk[row*pages+i]) : 0u;
    }
    Sort(scratch).SortDescendingBlockedToStriped(key);
    const int i=threadIdx.x;
    if(i<=budget) top[i]=__uint_as_float(key[0]);
    __syncthreads();
    const bool close=i>0 && i<budget && top[i]-top[i+1]<=4.e-5f*top[i];
    if(__syncthreads_or(close) && i==0) rows[atomicAdd(count,1)]=row;
}
extern "C" void mark_close(const float* risk,const long long* own,int* rows,int* count,
                           int n,int pages,cudaStream_t stream) {
    cudaMemsetAsync(count,0,sizeof(int),stream);
    if(pages<=256) close_rows<2><<<n,128,0,stream>>>(risk,own,rows,count,pages,pages/10);
    else close_rows<4><<<n,128,0,stream>>>(risk,own,rows,count,pages,pages/10);
}

// CUB's stable sort retains the lower page ID for equal keys.
template<int Items>
__global__ void sort_pages(const float* risk, const long long* own,
                           int* ids, bool* mask, int* rows, int* count, int p) {
    constexpr int Threads=64;
    using Sort = cub::BlockRadixSort<unsigned,Threads,Items,int>;
    __shared__ typename Sort::TempStorage storage;
    __shared__ unsigned sorted_top[45];
    int row = blockIdx.x;
    int k = p/10;
    unsigned key[Items];
    int page[Items];
    int own_page = own[row];
    #pragma unroll
    for (int j=0; j<Items; ++j) {
        int i = threadIdx.x*Items+j;
        unsigned bits = i==own_page ? 0x7f800000U : i<p ? __float_as_uint(risk[row*p+i]) : 0U;
        key[j] = i<p ? (bits>>31 ? ~bits : bits^0x80000000U) : 0U;
        page[j] = i;
    }
    Sort(storage).SortDescendingBlockedToStriped(key,page);
    #pragma unroll
    for (int j=0; j<Items; ++j) {
        int rank=threadIdx.x+j*Threads;
        if (rank<k) ids[row*k+rank]=page[j];
        if (page[j]<p) mask[row*p+page[j]]=rank<k;
    }
    {
        if (threadIdx.x<=k) sorted_top[threadIdx.x]=key[0];
        __syncthreads();
        bool close=false;
        if (threadIdx.x>0 && threadIdx.x<k) {
            // Risk scores are nonnegative on the selective recomputation path.
            float a=__uint_as_float(sorted_top[threadIdx.x]^0x80000000U);
            float b=__uint_as_float(sorted_top[threadIdx.x+1]^0x80000000U);
            close=a-b<=4.e-5f*a;
        }
        if (__syncthreads_or(close) && threadIdx.x==0) rows[atomicAdd(count,1)]=row;
    }
}
extern "C" void select_close(const float* risk,const long long* own,int* ids,bool* mask,
                             int* rows,int* count,int n,int p,cudaStream_t stream) {
    cudaMemsetAsync(count,0,sizeof(int),stream);
    if (p<=256) sort_pages<4><<<n,64,0,stream>>>(risk,own,ids,mask,rows,count,p);
    else sort_pages<8><<<n,64,0,stream>>>(risk,own,ids,mask,rows,count,p);
}
