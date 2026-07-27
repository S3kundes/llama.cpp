#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_top_k_stable_pairs(
        ggml_backend_cuda_context & ctx,
        const float * scores,
        const int32_t * indices,
        int64_t ncols,
        int64_t nrows,
        int64_t k,
        int32_t * dst);
