#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_sharded(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, int rank, int n_ranks, int64_t page_size);
