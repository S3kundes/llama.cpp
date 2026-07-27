#include "argsort.cuh"
#include "top-k.cuh"

#include <climits>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}

#ifdef GGML_CUDA_USE_CUB
static __global__ void top_k_pair_keys(
        const float * scores, const int32_t * indices, uint64_t * keys, int64_t ncols) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= ncols) {
        return;
    }

    const uint32_t bits = __float_as_uint(scores[i]);
    const uint32_t ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    keys[i] = (uint64_t(ordered) << 32) | ~uint32_t(indices[i]);
}

static __global__ void top_k_pair_indices(const uint64_t * keys, int32_t * dst, int64_t k) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < k) {
        dst[i] = int32_t(~uint32_t(keys[i]));
    }
}
#endif

bool ggml_cuda_top_k_stable_pairs(
        ggml_backend_cuda_context & ctx,
        const float * scores,
        const int32_t * indices,
        int64_t ncols,
        int64_t nrows,
        int64_t k,
        int32_t * dst) {
#ifdef GGML_CUDA_USE_CUB
    GGML_ASSERT(ncols >= k && k > 0 && ncols <= INT_MAX);
    const int ncols_i = (int) ncols;
    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<uint64_t> keys_in(pool, ncols);
    ggml_cuda_pool_alloc<uint64_t> keys_out(pool, ncols);

    size_t temp_size = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeysDescending(
            nullptr, temp_size, keys_in.get(), keys_out.get(), ncols_i, 0, 64, stream));
    ggml_cuda_pool_alloc<uint8_t> temp(pool, temp_size);

    const int threads = 256;
    const int blocks_keys = (ncols + threads - 1)/threads;
    const int blocks_dst = (k + threads - 1)/threads;
    for (int64_t row = 0; row < nrows; ++row) {
        top_k_pair_keys<<<blocks_keys, threads, 0, stream>>>(
                scores + row*ncols, indices + row*ncols, keys_in.get(), ncols);
        CUDA_CHECK(cub::DeviceRadixSort::SortKeysDescending(
                temp.get(), temp_size, keys_in.get(), keys_out.get(), ncols_i, 0, 64, stream));
        top_k_pair_indices<<<blocks_dst, threads, 0, stream>>>(keys_out.get(), dst + row*k, k);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(scores);
    GGML_UNUSED(indices);
    GGML_UNUSED(ncols);
    GGML_UNUSED(nrows);
    GGML_UNUSED(k);
    GGML_UNUSED(dst);
    return false;
#endif
}
