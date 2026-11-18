#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

// CUB's stable sort retains the lower page ID for equal keys.
template<int Threads, int Items, bool Flags>
__global__ void sort_pages(const float* risk, const long long* own,
                           int* ids, bool* mask, int* rows, int* count, int p) {
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
    if (Flags) {
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
template<int Threads, bool Flags>
void launch(const float* risk,const long long* own,int* ids,bool* mask,int* rows,int* count,
            int n,int p,cudaStream_t stream) {
    if (p<=256) sort_pages<Threads,256/Threads,Flags><<<n,Threads,0,stream>>>(risk,own,ids,mask,rows,count,p);
    else sort_pages<Threads,512/Threads,Flags><<<n,Threads,0,stream>>>(risk,own,ids,mask,rows,count,p);
}
extern "C" void select_pages(const float* risk,const long long* own,int* ids,bool* mask,int* rows,int* count,
                             int n,int p,int threads,bool flags,cudaStream_t stream) {
    if (flags) cudaMemsetAsync(count,0,sizeof(int),stream);
    if (threads==64) {
        if (flags) launch<64,true>(risk,own,ids,mask,rows,count,n,p,stream);
        else launch<64,false>(risk,own,ids,mask,rows,count,n,p,stream);
    } else {
        if (flags) launch<128,true>(risk,own,ids,mask,rows,count,n,p,stream);
        else launch<128,false>(risk,own,ids,mask,rows,count,n,p,stream);
    }
}
