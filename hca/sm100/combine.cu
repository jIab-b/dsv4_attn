#include "combine.h"

#include <math_constants.h>

namespace dsv4::hca::sm100 {

namespace {

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int HEAD_DIM_V, int BLOCK_HEADS, int MAX_SPLITS, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void hca_combine_kernel(__grid_constant__ const HcaCombineParams p) {
    static_assert(HEAD_DIM_V == 512);
    static_assert(NUM_THREADS / 32 == BLOCK_HEADS);
    static_assert(HEAD_DIM_V % (32 * 4) == 0);

    const int batch = blockIdx.x;
    const int head_block = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int head = head_block * BLOCK_HEADS + warp;
    if (batch >= p.B || head >= p.H) return;
    if (p.num_splits > MAX_SPLITS) return;

    __shared__ float split_scale[BLOCK_HEADS][MAX_SPLITS];

    constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / (32 * 4);
    constexpr int LSE_PER_THREAD = (MAX_SPLITS + 31) / 32;

    const float* lse_base =
        p.partial_lse
        + int64_t(batch) * p.stride_partial_lse_b
        + head;

    float local_lse[LSE_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < LSE_PER_THREAD; ++i) {
        const int split = i * 32 + lane;
        local_lse[i] = split < p.num_splits
            ? lse_base[int64_t(split) * p.stride_partial_lse_split]
            : -CUDART_INF_F;
    }

    float max_lse = -CUDART_INF_F;
    #pragma unroll
    for (int i = 0; i < LSE_PER_THREAD; ++i) {
        max_lse = fmaxf(max_lse, local_lse[i]);
    }
    max_lse = warp_reduce_max(max_lse);

    float sum_lse = 0.0f;
    if (max_lse != -CUDART_INF_F) {
        #pragma unroll
        for (int i = 0; i < LSE_PER_THREAD; ++i) {
            sum_lse += exp2f(local_lse[i] - max_lse);
        }
    }
    sum_lse = warp_reduce_sum(sum_lse);

    const bool has_tokens = sum_lse != 0.0f;
    const float global_lse = has_tokens ? log2f(sum_lse) + max_lse : -CUDART_INF_F;

    if (lane == 0 && p.lse != nullptr) {
        p.lse[int64_t(batch) * p.stride_lse_b + head] =
            has_tokens ? global_lse * CUDART_LN2_F : -CUDART_INF_F;
    }

    float denom_lse = global_lse;
    if (p.attn_sink != nullptr) {
        const float sink_lse = __ldg(p.attn_sink + head) * CUDART_L2E_F;
        denom_lse = has_tokens
            ? global_lse + log2f(1.0f + exp2f(sink_lse - global_lse))
            : sink_lse;
    }

    #pragma unroll
    for (int i = 0; i < LSE_PER_THREAD; ++i) {
        const int split = i * 32 + lane;
        if (split < MAX_SPLITS) {
            const float lse = local_lse[i];
            split_scale[warp][split] =
                (lse == -CUDART_INF_F || denom_lse == -CUDART_INF_F)
                    ? 0.0f
                    : exp2f(lse - denom_lse);
        }
    }
    __syncwarp();

    float4 accum[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        accum[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    for (int split = 0; split < p.num_splits; ++split) {
        const float scale = split_scale[warp][split];
        const float* src =
            p.partial_O
            + int64_t(split) * p.stride_partial_O_split
            + int64_t(batch) * p.stride_partial_O_b
            + int64_t(head)  * p.stride_partial_O_h
            + lane * 4;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const float4 v = *reinterpret_cast<const float4*>(src + i * 128);
            accum[i].x = fmaf(scale, v.x, accum[i].x);
            accum[i].y = fmaf(scale, v.y, accum[i].y);
            accum[i].z = fmaf(scale, v.z, accum[i].z);
            accum[i].w = fmaf(scale, v.w, accum[i].w);
        }
    }

    __nv_bfloat16* dst =
        p.O
        + int64_t(batch) * p.stride_O_b
        + int64_t(head)  * p.stride_O_h
        + lane * 4;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        __nv_bfloat16 out[4];
        out[0] = __float2bfloat16(accum[i].x);
        out[1] = __float2bfloat16(accum[i].y);
        out[2] = __float2bfloat16(accum[i].z);
        out[3] = __float2bfloat16(accum[i].w);
        *reinterpret_cast<uint64_t*>(dst + i * 128) =
            *reinterpret_cast<uint64_t*>(out);
    }
}

template<int MAX_SPLITS>
void launch_hca_combine_max_splits(const HcaCombineParams& p, cudaStream_t stream) {
    constexpr int HEAD_DIM_V = 512;
    constexpr int BLOCK_HEADS = 8;
    constexpr int NUM_THREADS = BLOCK_HEADS * 32;
    dim3 grid(p.B, (p.H + BLOCK_HEADS - 1) / BLOCK_HEADS);
    hca_combine_kernel<HEAD_DIM_V, BLOCK_HEADS, MAX_SPLITS, NUM_THREADS>
        <<<grid, NUM_THREADS, 0, stream>>>(p);
}

}  // namespace

void launch_hca_combine(const HcaCombineParams& p, cudaStream_t stream) {
    if (p.B <= 0 || p.H <= 0 || p.num_splits <= 0) return;
    if (p.num_splits <= 32) {
        launch_hca_combine_max_splits<32>(p, stream);
    } else if (p.num_splits <= 64) {
        launch_hca_combine_max_splits<64>(p, stream);
    } else if (p.num_splits <= 96) {
        launch_hca_combine_max_splits<96>(p, stream);
    } else if (p.num_splits <= 128) {
        launch_hca_combine_max_splits<128>(p, stream);
    } else if (p.num_splits <= 160) {
        launch_hca_combine_max_splits<160>(p, stream);
    } else if (p.num_splits <= 256) {
        launch_hca_combine_max_splits<256>(p, stream);
    } else if (p.num_splits <= 512) {
        launch_hca_combine_max_splits<512>(p, stream);
    } else if (p.num_splits <= 1024) {
        launch_hca_combine_max_splits<1024>(p, stream);
    }
}

}  // namespace dsv4::hca::sm100
