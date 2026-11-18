#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

template<int Threads, int Items>
__global__ void sort_rows(const float* scores, int* order) {
    using Sort = cub::BlockRadixSort<unsigned, Threads, Items, int>;
    __shared__ typename Sort::TempStorage temp;
    constexpr int size = Threads * Items;
    const int base = blockIdx.x * size;
    unsigned keys[Items];
    int indices[Items];
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = threadIdx.x * Items + j;
        const float x = scores[base + i];
        const unsigned bits = __float_as_uint(x == 0.f ? 0.f : x);
        keys[j] = isnan(x) ? ~0u : bits ^ ((bits >> 31) ? ~0u : 0x80000000u);
        indices[j] = i;
    }
    Sort(temp).SortBlockedToStriped(keys, indices);
    #pragma unroll
    for (int j = 0; j < Items; ++j)
        order[base + j * Threads + threadIdx.x] = indices[j];
}

extern "C" void group_radix(const float* scores, int* order, int groups, int size, cudaStream_t stream) {
    if (size == 1536) sort_rows<256, 6><<<groups, 256, 0, stream>>>(scores, order);
    else sort_rows<512, 9><<<groups, 512, 0, stream>>>(scores, order);
}
