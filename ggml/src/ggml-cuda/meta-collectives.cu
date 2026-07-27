#include "meta-collectives.cuh"

#ifdef GGML_USE_NCCL

#include "fattn.cuh"
#include "lightning-indexer.cuh"
#include "set-rows.cuh"
#include "top-k.cuh"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

struct meta_sequence_layout {
    int n_segments;
    int n_ranks;
    int rank;
    int64_t global_start[16];
    int64_t local_start[16];
    int64_t page_size[16];
    uint32_t n_repeat[16];
};

struct meta_head_layout {
    int n_ranks;
    int64_t n_head;
    int64_t head_start[GGML_BACKEND_META_MAX_DEVICES];
    int64_t head_count[GGML_BACKEND_META_MAX_DEVICES];
    int64_t q_offset[GGML_BACKEND_META_MAX_DEVICES];
    int64_t elem_offset[GGML_BACKEND_META_MAX_DEVICES];
    int64_t row_offset[GGML_BACKEND_META_MAX_DEVICES];
};

static ggml_backend_cuda_context & get_cuda_context(ggml_backend_t backend) {
    return *(ggml_backend_cuda_context *) backend->context;
}

static void set_contiguous_tensor(
        ggml_tensor & tensor,
        ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        void * data) {
    tensor.type = type;
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = ne3;
    tensor.nb[0] = ggml_type_size(type);
    tensor.nb[1] = tensor.nb[0]*(ne0/ggml_blck_size(type));
    tensor.nb[2] = tensor.nb[1]*ne1;
    tensor.nb[3] = tensor.nb[2]*ne2;
    tensor.data = data;
}

static meta_sequence_layout make_sequence_layout(
        const ggml_backend_meta_split_state & split,
        int n_ranks,
        int rank) {
    GGML_ASSERT(split.n_segments > 0 && split.n_segments <= 16);

    meta_sequence_layout layout = {};
    layout.n_segments = split.n_segments;
    layout.n_ranks = n_ranks;
    layout.rank = rank;

    int64_t global_start = 0;
    int64_t local_start = 0;
    for (int s = 0; s < layout.n_segments; ++s) {
        const int64_t page_size = split.ne[s*n_ranks];
        GGML_ASSERT(page_size > 0);
        for (int r = 1; r < n_ranks; ++r) {
            GGML_ASSERT(split.ne[s*n_ranks + r] == page_size);
        }

        layout.global_start[s] = global_start;
        layout.local_start[s] = local_start;
        layout.page_size[s] = page_size;
        layout.n_repeat[s] = split.nr[s];
        global_start += page_size*n_ranks*split.nr[s];
        local_start += page_size*split.nr[s];
    }

    return layout;
}

static meta_head_layout make_head_layout(
        const ggml_backend_meta_split_state & split,
        int n_ranks,
        int64_t n_query,
        int64_t n_stream,
        int64_t q_dim,
        int64_t value_dim) {
    GGML_ASSERT(split.n_segments == 1 && split.nr[0] == 1);

    meta_head_layout layout = {};
    layout.n_ranks = n_ranks;
    int64_t n_head = 0;
    int64_t q_offset = 0;
    int64_t elem_offset = 0;
    int64_t row_offset = 0;
    for (int r = 0; r < n_ranks; ++r) {
        layout.head_start[r] = n_head;
        layout.head_count[r] = split.ne[r];
        layout.q_offset[r] = q_offset;
        layout.elem_offset[r] = elem_offset;
        layout.row_offset[r] = row_offset;
        n_head += split.ne[r];
        q_offset += split.ne[r]*n_query*n_stream*q_dim;
        elem_offset += split.ne[r]*n_query*n_stream*value_dim;
        row_offset += split.ne[r]*n_query*n_stream;
    }
    layout.n_head = n_head;
    return layout;
}

static __device__ __forceinline__ int64_t sequence_local_to_global(
        const meta_sequence_layout & layout, int64_t local_index) {
    for (int s = 0; s < layout.n_segments; ++s) {
        const int64_t n_local = layout.page_size[s]*layout.n_repeat[s];
        if (local_index < layout.local_start[s] || local_index >= layout.local_start[s] + n_local) {
            continue;
        }
        const int64_t relative = local_index - layout.local_start[s];
        const int64_t repeat = relative/layout.page_size[s];
        const int64_t in_page = relative%layout.page_size[s];
        return layout.global_start[s] +
                (repeat*layout.n_ranks + layout.rank)*layout.page_size[s] + in_page;
    }
    return -1;
}

static __global__ void pack_q(
        const char * src,
        float * dst,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        size_t nb0,
        size_t nb1,
        size_t nb2,
        size_t nb3) {
    const int64_t n = ne0*ne1*ne2*ne3;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int64_t tmp = i;
    const int64_t i0 = tmp%ne0;
    tmp /= ne0;
    const int64_t i1 = tmp%ne1;
    tmp /= ne1;
    const int64_t i2 = tmp%ne2;
    const int64_t i3 = tmp/ne2;
    dst[i] = *(const float *) (src + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
}

static __global__ void unpack_q(
        const float * gathered,
        float * dst,
        meta_head_layout layout,
        int64_t ne0,
        int64_t ne1,
        int64_t ne3) {
    const int64_t n = ne0*ne1*layout.n_head*ne3;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int64_t tmp = i;
    const int64_t i0 = tmp%ne0;
    tmp /= ne0;
    const int64_t i1 = tmp%ne1;
    tmp /= ne1;
    const int64_t head = tmp%layout.n_head;
    const int64_t stream = tmp/layout.n_head;

    int owner = 0;
    while (owner + 1 < layout.n_ranks && head >= layout.head_start[owner] + layout.head_count[owner]) {
        ++owner;
    }
    const int64_t local_head = head - layout.head_start[owner];
    const int64_t src_index = layout.q_offset[owner] +
            ((stream*layout.head_count[owner] + local_head)*ne1 + i1)*ne0 + i0;
    dst[i] = gathered[src_index];
}

static __global__ void pack_attention_parts(
        const float * src,
        const float2 * src_meta,
        float * dst,
        float2 * dst_meta,
        meta_head_layout layout,
        int64_t value_dim,
        int64_t n_query,
        int64_t n_stream) {
    const int64_t n = value_dim*layout.n_head*n_query*n_stream;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int64_t tmp = i;
    const int64_t d = tmp%value_dim;
    tmp /= value_dim;
    const int64_t head = tmp%layout.n_head;
    tmp /= layout.n_head;
    const int64_t query = tmp%n_query;
    const int64_t stream = tmp/n_query;

    int owner = 0;
    while (owner + 1 < layout.n_ranks && head >= layout.head_start[owner] + layout.head_count[owner]) {
        ++owner;
    }
    const int64_t local_head = head - layout.head_start[owner];
    const int64_t local_row = (stream*n_query + query)*layout.head_count[owner] + local_head;
    dst[layout.elem_offset[owner] + local_row*value_dim + d] = src[i];
    if (d == 0) {
        const int64_t global_row = (stream*n_query + query)*layout.n_head + head;
        dst_meta[layout.row_offset[owner] + local_row] = src_meta[global_row];
    }
}

static __global__ void extract_attention_max(
        const float2 * src,
        float * dst,
        int64_t n_rows) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n_rows) {
        return;
    }
    dst[i] = src[i].y > 0.0f ? src[i].x : -FLT_MAX;
}

static __global__ void pack_attention_reduction(
        const float * src,
        const float2 * src_meta,
        const float * global_max,
        float * dst,
        int64_t n_head,
        int64_t heads_per_rank,
        int64_t value_dim,
        int64_t n_query,
        int64_t n_stream) {
    const int64_t n = value_dim*n_head*n_query*n_stream;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int64_t tmp = i;
    const int64_t d = tmp%value_dim;
    tmp /= value_dim;
    const int64_t head = tmp%n_head;
    tmp /= n_head;
    const int64_t query = tmp%n_query;
    const int64_t stream = tmp/n_query;

    const int64_t owner = head/heads_per_rank;
    const int64_t local_head = head%heads_per_rank;
    const int64_t local_rows = heads_per_rank*n_query*n_stream;
    const int64_t local_row = (stream*n_query + query)*heads_per_rank + local_head;
    const int64_t global_row = (stream*n_query + query)*n_head + head;
    const float2 meta = src_meta[global_row];
    const float scale = meta.y > 0.0f ? expf(meta.x - global_max[global_row]) : 0.0f;
    const int64_t dst_row = owner*local_rows + local_row;
    dst[dst_row*(value_dim + 1) + d] = scale*src[i];
    if (d == 0) {
        dst[dst_row*(value_dim + 1) + value_dim] = scale*meta.y;
    }
}

static __global__ void normalize_attention_reduction(
        const float * src,
        float * dst,
        int64_t value_dim,
        int64_t n_rows) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= value_dim*n_rows) {
        return;
    }
    const int64_t row = i/value_dim;
    const int64_t d = i%value_dim;
    const float denominator = src[row*(value_dim + 1) + value_dim];
    dst[i] = denominator > 0.0f ? src[row*(value_dim + 1) + d]/denominator : 0.0f;
}

static __global__ void combine_attention_parts(
        const float * parts,
        const float2 * metas,
        float * dst,
        int n_ranks,
        int64_t value_dim,
        int64_t n_rows) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= value_dim*n_rows) {
        return;
    }

    const int64_t row = i/value_dim;
    const int64_t d = i%value_dim;
    float max_value = -FLT_MAX;
    for (int r = 0; r < n_ranks; ++r) {
        const float2 meta = metas[r*n_rows + row];
        if (meta.y > 0.0f) {
            max_value = fmaxf(max_value, meta.x);
        }
    }

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int r = 0; r < n_ranks; ++r) {
        const float2 meta = metas[r*n_rows + row];
        if (meta.y <= 0.0f) {
            continue;
        }
        const float scale = expf(meta.x - max_value);
        numerator += scale*parts[(r*n_rows + row)*value_dim + d];
        denominator += scale*meta.y;
    }
    dst[i] = denominator > 0.0f ? numerator/denominator : 0.0f;
}

static __global__ void gather_top_k_candidates(
        const float * scores,
        const int32_t * local_indices,
        float * candidate_scores,
        int32_t * candidate_indices,
        meta_sequence_layout layout,
        int64_t local_k,
        int64_t n_candidates,
        int64_t n_rows) {
    const int64_t n = n_candidates*n_rows;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t row = i/n_candidates;
    const int32_t local = local_indices[i];
    if (local < 0 || local >= local_k) {
        return;
    }
    candidate_scores[i] = scores[row*local_k + local];
    candidate_indices[i] = (int32_t) sequence_local_to_global(layout, local);
}

static __global__ void reorder_top_k_candidates(
        const float * scores_in,
        const int32_t * indices_in,
        float * scores_out,
        int32_t * indices_out,
        int n_ranks,
        int64_t n_candidates,
        int64_t n_rows) {
    const int64_t n = n_ranks*n_candidates*n_rows;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int64_t tmp = i;
    const int64_t candidate = tmp%n_candidates;
    tmp /= n_candidates;
    const int rank = tmp%n_ranks;
    const int64_t row = tmp/n_ranks;
    const int64_t src = (rank*n_rows + row)*n_candidates + candidate;
    const int64_t dst = row*(n_ranks*n_candidates) + rank*n_candidates + candidate;
    scores_out[dst] = scores_in[src];
    indices_out[dst] = indices_in[src];
}

template<typename Kernel, typename... Args>
static void launch_1d(int64_t n, cudaStream_t stream, Kernel kernel, Args... args) {
    const int threads = 256;
    const int blocks = (n + threads - 1)/threads;
    const ggml_cuda_kernel_launch_params params(dim3(blocks), dim3(threads), 0, stream);
    ggml_cuda_kernel_launch(kernel, params, args...);
}

static bool execute_set_rows(
        ggml_backend_t * backends,
        size_t n_backends,
        const ggml_backend_comm_graph_node * graph_node) {
    const auto & split = graph_node->src[2];
    GGML_ASSERT(split.axis == GGML_BACKEND_SPLIT_AXIS_1 && split.n_segments == 1);
    const int64_t page_size = split.ne[0];
    for (size_t r = 0; r < n_backends; ++r) {
        GGML_ASSERT(split.ne[r] == page_size);
        ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
        ggml_cuda_set_device(ctx.device);
        ggml_cuda_op_set_rows_sharded(ctx, graph_node->nodes[r], r, n_backends, page_size);
    }
    return true;
}

static bool execute_lightning_indexer(
        ggml_backend_t * backends,
        size_t n_backends,
        const ggml_backend_comm_graph_node * graph_node) {
    for (size_t r = 0; r < n_backends; ++r) {
        ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
        ggml_cuda_set_device(ctx.device);
        ggml_tensor * node = graph_node->nodes[r];
        GGML_ASSERT(node->src[3]->ne[0] == node->src[1]->ne[2]);
        ggml_cuda_lightning_indexer(ctx, node);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool execute_top_k(
        ggml_backend_t * backends,
        ncclComm_t * comms,
        size_t n_backends,
        const ggml_backend_comm_graph_node * graph_node) {
    const int64_t local_k = graph_node->nodes[0]->src[0]->ne[0];
    const int64_t n_rows = ggml_nrows(graph_node->nodes[0]->src[0]);
    const int64_t k = graph_node->nodes[0]->ne[0];
    const int64_t n_candidates = std::min(local_k, k);
    const int64_t n_candidates_all = n_candidates*n_backends;
    GGML_ASSERT(n_candidates_all >= k);

    std::array<ggml_cuda_pool_alloc<int32_t>, GGML_CUDA_MAX_DEVICES> local_indices;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> candidate_scores;
    std::array<ggml_cuda_pool_alloc<int32_t>, GGML_CUDA_MAX_DEVICES> candidate_indices;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> gathered_scores;
    std::array<ggml_cuda_pool_alloc<int32_t>, GGML_CUDA_MAX_DEVICES> gathered_indices;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> ordered_scores;
    std::array<ggml_cuda_pool_alloc<int32_t>, GGML_CUDA_MAX_DEVICES> ordered_indices;
    std::array<ggml_tensor, GGML_CUDA_MAX_DEVICES> local_nodes;

    for (size_t r = 0; r < n_backends; ++r) {
        ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
        ggml_cuda_set_device(ctx.device);
        ggml_tensor * node = graph_node->nodes[r];
        GGML_ASSERT(node->src[0]->ne[0] == local_k && ggml_nrows(node->src[0]) == n_rows);

        int32_t * local_data = local_indices[r].alloc(ctx.pool(), n_candidates*n_rows);
        memset(&local_nodes[r], 0, sizeof(local_nodes[r]));
        set_contiguous_tensor(local_nodes[r], GGML_TYPE_I32,
                n_candidates, node->ne[1], node->ne[2], node->ne[3], local_data);
        local_nodes[r].op = GGML_OP_TOP_K;
        local_nodes[r].src[0] = node->src[0];
        ggml_cuda_op_top_k(ctx, &local_nodes[r]);

        float * score_data = candidate_scores[r].alloc(ctx.pool(), n_candidates*n_rows);
        int32_t * index_data = candidate_indices[r].alloc(ctx.pool(), n_candidates*n_rows);
        const meta_sequence_layout layout = make_sequence_layout(graph_node->src[0], n_backends, r);
        launch_1d(n_candidates*n_rows, ctx.stream(), gather_top_k_candidates,
                (const float *) node->src[0]->data, local_data, score_data, index_data,
                layout, local_k, n_candidates, n_rows);

        gathered_scores[r].alloc(ctx.pool(), n_candidates_all*n_rows);
        gathered_indices[r].alloc(ctx.pool(), n_candidates_all*n_rows);
        ordered_scores[r].alloc(ctx.pool(), n_candidates_all*n_rows);
        ordered_indices[r].alloc(ctx.pool(), n_candidates_all*n_rows);
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t r = 0; r < n_backends; ++r) {
        ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
        ggml_cuda_set_device(ctx.device);
        NCCL_CHECK(ncclAllGather(candidate_scores[r].get(), gathered_scores[r].get(),
                n_candidates*n_rows, ncclFloat, comms[r], ctx.stream()));
        NCCL_CHECK(ncclAllGather(candidate_indices[r].get(), gathered_indices[r].get(),
                n_candidates*n_rows, ncclInt32, comms[r], ctx.stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t r = 0; r < n_backends; ++r) {
        ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
        ggml_cuda_set_device(ctx.device);
        launch_1d(n_candidates_all*n_rows, ctx.stream(), reorder_top_k_candidates,
                gathered_scores[r].get(), gathered_indices[r].get(),
                ordered_scores[r].get(), ordered_indices[r].get(),
                n_backends, n_candidates, n_rows);
        if (!ggml_cuda_top_k_stable_pairs(ctx,
                ordered_scores[r].get(), ordered_indices[r].get(),
                n_candidates_all, n_rows, k, (int32_t *) graph_node->nodes[r]->data)) {
            return false;
        }
    }
    return true;
}

static bool execute_flash_attn(
        ggml_backend_t * backends,
        ncclComm_t * comms,
        size_t n_backends,
        const ggml_backend_comm_graph_node * graph_node) {
    ggml_tensor * node0 = graph_node->nodes[0];
    const int64_t q_dim = node0->src[0]->ne[0];
    const int64_t n_query = node0->src[0]->ne[1];
    const int64_t n_stream = node0->src[0]->ne[3];
    const int64_t value_dim = node0->ne[0];
    const meta_head_layout heads = make_head_layout(
            graph_node->src[0], n_backends, n_query, n_stream, q_dim, value_dim);
    const int64_t n_head = heads.n_head;
    const int64_t q_elements = q_dim*n_query*n_head*n_stream;
    const int64_t out_elements = value_dim*n_head*n_query*n_stream;
    const int64_t out_rows = n_head*n_query*n_stream;
    bool equal_heads = heads.head_count[0] > 0;
    for (size_t r = 1; r < n_backends; ++r) {
        equal_heads = equal_heads && heads.head_count[r] == heads.head_count[0];
    }

    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> q_local;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> q_gathered;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> q_full;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> sinks_full;
    std::array<ggml_cuda_pool_alloc<uint8_t>, GGML_CUDA_MAX_DEVICES> fa_storage;
    std::array<ggml_cuda_pool_alloc<float2>, GGML_CUDA_MAX_DEVICES> fa_meta;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> global_max;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> packed_reduction;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> reduced_reduction;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> packed_parts;
    std::array<ggml_cuda_pool_alloc<float2>, GGML_CUDA_MAX_DEVICES> packed_meta;
    std::array<ggml_cuda_pool_alloc<float>, GGML_CUDA_MAX_DEVICES> received_parts;
    std::array<ggml_cuda_pool_alloc<float2>, GGML_CUDA_MAX_DEVICES> received_meta;

    std::array<ggml_tensor, GGML_CUDA_MAX_DEVICES> q_tensors;
    std::array<ggml_tensor, GGML_CUDA_MAX_DEVICES> sink_tensors;
    std::array<ggml_tensor, GGML_CUDA_MAX_DEVICES> fa_nodes;

    std::array<int64_t, GGML_CUDA_MAX_DEVICES> q_count = {};
    std::array<int64_t, GGML_CUDA_MAX_DEVICES> q_offset = {};
    int64_t q_total = 0;
    const bool has_sinks = node0->src[4] != nullptr;

    for (size_t r = 0; r < n_backends; ++r) {
        ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
        ggml_cuda_set_device(ctx.device);
        ggml_tensor * q = graph_node->nodes[r]->src[0];
        q_count[r] = ggml_nelements(q);
        q_offset[r] = q_total;
        q_total += q_count[r];
        float * packed = q_local[r].alloc(ctx.pool(), q_count[r]);
        launch_1d(q_count[r], ctx.stream(), pack_q,
                (const char *) q->data, packed, q->ne[0], q->ne[1], q->ne[2], q->ne[3],
                q->nb[0], q->nb[1], q->nb[2], q->nb[3]);
        q_gathered[r].alloc(ctx.pool(), q_elements);
        if (n_stream > 1) {
            q_full[r].alloc(ctx.pool(), q_elements);
        }
        if (has_sinks) {
            sinks_full[r].alloc(ctx.pool(), n_head);
        }
    }
    GGML_ASSERT(q_total == q_elements);

    if (equal_heads) {
        NCCL_CHECK(ncclGroupStart());
        for (size_t r = 0; r < n_backends; ++r) {
            ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
            ggml_cuda_set_device(ctx.device);
            NCCL_CHECK(ncclAllGather(q_local[r].get(), q_gathered[r].get(),
                    q_count[r], ncclFloat, comms[r], ctx.stream()));
        }
        if (has_sinks) {
            for (size_t r = 0; r < n_backends; ++r) {
                ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
                ggml_cuda_set_device(ctx.device);
                NCCL_CHECK(ncclAllGather(graph_node->nodes[r]->src[4]->data, sinks_full[r].get(),
                        heads.head_count[r], ncclFloat, comms[r], ctx.stream()));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
    } else {
        for (size_t dst = 0; dst < n_backends; ++dst) {
            ggml_backend_cuda_context & ctx = get_cuda_context(backends[dst]);
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaMemcpyAsync(q_gathered[dst].get() + q_offset[dst], q_local[dst].get(),
                    q_count[dst]*sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
            if (has_sinks) {
                CUDA_CHECK(cudaMemcpyAsync(sinks_full[dst].get() + heads.head_start[dst],
                        graph_node->nodes[dst]->src[4]->data,
                        heads.head_count[dst]*sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
            }
        }

        NCCL_CHECK(ncclGroupStart());
        for (size_t src = 0; src < n_backends; ++src) {
            for (size_t dst = 0; dst < n_backends; ++dst) {
                if (src == dst) {
                    continue;
                }
                ggml_backend_cuda_context & src_ctx = get_cuda_context(backends[src]);
                ggml_backend_cuda_context & dst_ctx = get_cuda_context(backends[dst]);
                ggml_cuda_set_device(src_ctx.device);
                NCCL_CHECK(ncclSend(q_local[src].get(), q_count[src], ncclFloat, dst, comms[src], src_ctx.stream()));
                if (has_sinks) {
                    NCCL_CHECK(ncclSend(graph_node->nodes[src]->src[4]->data,
                            heads.head_count[src], ncclFloat, dst, comms[src], src_ctx.stream()));
                }
                ggml_cuda_set_device(dst_ctx.device);
                NCCL_CHECK(ncclRecv(q_gathered[dst].get() + q_offset[src], q_count[src],
                        ncclFloat, src, comms[dst], dst_ctx.stream()));
                if (has_sinks) {
                    NCCL_CHECK(ncclRecv(sinks_full[dst].get() + heads.head_start[src], heads.head_count[src],
                            ncclFloat, src, comms[dst], dst_ctx.stream()));
                }
            }
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    for (size_t r = 0; r < n_backends; ++r) {
        ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
        ggml_cuda_set_device(ctx.device);
        ggml_tensor * node = graph_node->nodes[r];
        float * q_data = q_gathered[r].get();
        if (n_stream > 1) {
            q_data = q_full[r].get();
            launch_1d(q_elements, ctx.stream(), unpack_q,
                    q_gathered[r].get(), q_data, heads, q_dim, n_query, n_stream);
        }

        memset(&q_tensors[r], 0, sizeof(q_tensors[r]));
        set_contiguous_tensor(q_tensors[r], GGML_TYPE_F32,
                q_dim, n_query, n_head, n_stream, q_data);

        if (has_sinks) {
            memset(&sink_tensors[r], 0, sizeof(sink_tensors[r]));
            set_contiguous_tensor(sink_tensors[r], GGML_TYPE_F32,
                    n_head, 1, 1, 1, sinks_full[r].get());
        }

        const ggml_tensor * mask = node->src[3];
        GGML_ASSERT(mask != nullptr && mask->type == GGML_TYPE_F16);
        GGML_ASSERT(mask->ne[0] == node->src[1]->ne[1]);

        fa_nodes[r] = *node;
        set_contiguous_tensor(fa_nodes[r], GGML_TYPE_F32,
                value_dim, n_head, n_query, n_stream, nullptr);
        fa_nodes[r].src[0] = &q_tensors[r];
        fa_nodes[r].src[4] = has_sinks && r == 0 ? &sink_tensors[r] : nullptr;
        const size_t fa_size = ggml_cuda_flash_attn_ext_partial_get_alloc_size(ctx.device, &fa_nodes[r]);
        fa_nodes[r].data = fa_storage[r].alloc(ctx.pool(), fa_size);
        float2 * meta = fa_meta[r].alloc(ctx.pool(), out_rows);
        ggml_cuda_flash_attn_ext_partial(ctx, &fa_nodes[r], meta);

        if (equal_heads) {
            float * max_data = global_max[r].alloc(ctx.pool(), out_rows);
            launch_1d(out_rows, ctx.stream(), extract_attention_max, meta, max_data, out_rows);
        } else {
            float * packed_part = packed_parts[r].alloc(ctx.pool(), out_elements);
            float2 * packed_m = packed_meta[r].alloc(ctx.pool(), out_rows);
            launch_1d(out_elements, ctx.stream(), pack_attention_parts,
                    (const float *) fa_nodes[r].data, meta, packed_part, packed_m,
                    heads, value_dim, n_query, n_stream);

            const int64_t local_elements = value_dim*heads.head_count[r]*n_query*n_stream;
            const int64_t local_rows = heads.head_count[r]*n_query*n_stream;
            received_parts[r].alloc(ctx.pool(), n_backends*local_elements);
            received_meta[r].alloc(ctx.pool(), n_backends*local_rows);
            CUDA_CHECK(cudaMemcpyAsync(received_parts[r].get() + r*local_elements,
                    packed_part + heads.elem_offset[r], local_elements*sizeof(float),
                    cudaMemcpyDeviceToDevice, ctx.stream()));
            CUDA_CHECK(cudaMemcpyAsync(received_meta[r].get() + r*local_rows,
                    packed_m + heads.row_offset[r], local_rows*sizeof(float2),
                    cudaMemcpyDeviceToDevice, ctx.stream()));
        }
    }

    if (equal_heads) {
        NCCL_CHECK(ncclGroupStart());
        for (size_t r = 0; r < n_backends; ++r) {
            ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
            ggml_cuda_set_device(ctx.device);
            NCCL_CHECK(ncclAllReduce(global_max[r].get(), global_max[r].get(),
                    out_rows, ncclFloat, ncclMax, comms[r], ctx.stream()));
        }
        NCCL_CHECK(ncclGroupEnd());

        const int64_t heads_per_rank = heads.head_count[0];
        const int64_t local_rows = heads_per_rank*n_query*n_stream;
        const int64_t reduction_count = local_rows*(value_dim + 1);
        GGML_ASSERT(heads_per_rank*(int64_t) n_backends == n_head);
        for (size_t r = 0; r < n_backends; ++r) {
            ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
            ggml_cuda_set_device(ctx.device);
            float * packed = packed_reduction[r].alloc(ctx.pool(), n_backends*reduction_count);
            reduced_reduction[r].alloc(ctx.pool(), reduction_count);
            launch_1d(out_elements, ctx.stream(), pack_attention_reduction,
                    (const float *) fa_nodes[r].data, fa_meta[r].get(), global_max[r].get(), packed,
                    n_head, heads_per_rank, value_dim, n_query, n_stream);
        }

        NCCL_CHECK(ncclGroupStart());
        for (size_t r = 0; r < n_backends; ++r) {
            ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
            ggml_cuda_set_device(ctx.device);
            NCCL_CHECK(ncclReduceScatter(packed_reduction[r].get(), reduced_reduction[r].get(),
                    reduction_count, ncclFloat, ncclSum, comms[r], ctx.stream()));
        }
        NCCL_CHECK(ncclGroupEnd());

        for (size_t r = 0; r < n_backends; ++r) {
            ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
            ggml_cuda_set_device(ctx.device);
            launch_1d(value_dim*local_rows, ctx.stream(), normalize_attention_reduction,
                    reduced_reduction[r].get(), (float *) graph_node->nodes[r]->data,
                    value_dim, local_rows);
        }
    } else {
        NCCL_CHECK(ncclGroupStart());
        for (size_t src = 0; src < n_backends; ++src) {
            for (size_t dst = 0; dst < n_backends; ++dst) {
                if (src == dst) {
                    continue;
                }
                const int64_t dst_elements = value_dim*heads.head_count[dst]*n_query*n_stream;
                const int64_t dst_rows = heads.head_count[dst]*n_query*n_stream;
                ggml_backend_cuda_context & src_ctx = get_cuda_context(backends[src]);
                ggml_backend_cuda_context & dst_ctx = get_cuda_context(backends[dst]);
                ggml_cuda_set_device(src_ctx.device);
                NCCL_CHECK(ncclSend(packed_parts[src].get() + heads.elem_offset[dst], dst_elements,
                        ncclFloat, dst, comms[src], src_ctx.stream()));
                NCCL_CHECK(ncclSend(packed_meta[src].get() + heads.row_offset[dst], 2*dst_rows,
                        ncclFloat, dst, comms[src], src_ctx.stream()));
                ggml_cuda_set_device(dst_ctx.device);
                NCCL_CHECK(ncclRecv(received_parts[dst].get() + src*dst_elements, dst_elements,
                        ncclFloat, src, comms[dst], dst_ctx.stream()));
                NCCL_CHECK(ncclRecv(received_meta[dst].get() + src*dst_rows, 2*dst_rows,
                        ncclFloat, src, comms[dst], dst_ctx.stream()));
            }
        }
        NCCL_CHECK(ncclGroupEnd());

        for (size_t r = 0; r < n_backends; ++r) {
            ggml_backend_cuda_context & ctx = get_cuda_context(backends[r]);
            ggml_cuda_set_device(ctx.device);
            const int64_t local_rows = heads.head_count[r]*n_query*n_stream;
            launch_1d(value_dim*local_rows, ctx.stream(), combine_attention_parts,
                    received_parts[r].get(), received_meta[r].get(),
                    (float *) graph_node->nodes[r]->data, n_backends, value_dim, local_rows);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_meta_execute_graph_node(
        ggml_backend_t * backends,
        ncclComm_t * comms,
        size_t n_backends,
        const ggml_backend_comm_graph_node * node) {
    GGML_ASSERT(n_backends > 1 && n_backends <= GGML_BACKEND_META_MAX_DEVICES);
    switch (node->nodes[0]->op) {
        case GGML_OP_SET_ROWS:
            return execute_set_rows(backends, n_backends, node);
        case GGML_OP_LIGHTNING_INDEXER:
            return execute_lightning_indexer(backends, n_backends, node);
        case GGML_OP_TOP_K:
            return execute_top_k(backends, comms, n_backends, node);
        case GGML_OP_FLASH_ATTN_EXT:
            return execute_flash_attn(backends, comms, n_backends, node);
        default:
            return false;
    }
}

#endif
