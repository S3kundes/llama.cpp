#pragma once

#include "common.cuh"
#include "ggml-backend.h"

#ifdef GGML_USE_NCCL
bool ggml_cuda_meta_execute_graph_node(
        ggml_backend_t * backends,
        ncclComm_t * comms,
        size_t n_backends,
        const ggml_backend_comm_graph_node * node);
#endif
