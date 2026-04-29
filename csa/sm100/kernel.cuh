#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../params.h"

namespace dsv4::csa::sm100 {

// CSA Pass-2 reduce kernel.
//
// Implements Eqs 11-12 of DeepSeek-V4 §2.3.1.
//
// One CTA = one output slot (one compressed block).
// THREADS threads cooperate over c_out channels.
// Each thread handles ceil(c_out / THREADS) channels independently
// (per-channel softmax has no cross-channel dependency).
//
// For block 0: C_b is logically zero, Z_b is logically -inf. We branch
// on `block_idx == 0` to skip loading the (nonexistent) prev block and
// to fall back to a single-branch softmax (equivalent: weights on the
// b-side go to zero).
//
// Inputs are bf16, accumulators are fp32, output is bf16.
template<int M, int THREADS>
__global__ void csa_compress_reduce_kernel(CsaCompressReduceParams p) {
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");
    const int batch_idx = blockIdx.y;
    const int blk_idx   = blockIdx.x;
    if (blk_idx >= p.n_blocks) return;

    const int tid    = threadIdx.x;
    const int c_out  = p.c_out;

    const __nv_bfloat16* C_a_base = p.C_a + batch_idx * p.stride_C_b + blk_idx * M * p.stride_C_n;
    const __nv_bfloat16* Z_a_base = p.Z_a + batch_idx * p.stride_C_b + blk_idx * M * p.stride_C_n;
    const __nv_bfloat16* C_b_base = (blk_idx == 0) ? nullptr
                                  : p.C_b + batch_idx * p.stride_C_b + (blk_idx - 1) * M * p.stride_C_n;
    const __nv_bfloat16* Z_b_base = (blk_idx == 0) ? nullptr
                                  : p.Z_b + batch_idx * p.stride_C_b + (blk_idx - 1) * M * p.stride_C_n;

    __nv_bfloat16* out_base = p.out + batch_idx * p.stride_out_b + blk_idx * p.stride_out_blk;

    // Each thread strides through its assigned channels.
    for (int ch = tid; ch < c_out; ch += THREADS) {
        // Load 2M (or M, for block 0) gating logits and value rows.
        float z[2 * M];
        float c_val[2 * M];

        // a-side: positions 0..M-1
        #pragma unroll
        for (int j = 0; j < M; ++j) {
            float zaj = __bfloat162float(Z_a_base[j * p.stride_C_n + ch]);
            float baj = __bfloat162float(p.B_a[j * c_out + ch]);
            z[j]      = zaj + baj;
            c_val[j]  = __bfloat162float(C_a_base[j * p.stride_C_n + ch]);
        }
        // b-side: positions M..2M-1 (taken from prev block)
        if (blk_idx == 0) {
            #pragma unroll
            for (int j = 0; j < M; ++j) {
                z[M + j]     = -INFINITY;
                c_val[M + j] = 0.0f;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < M; ++j) {
                float zbj = __bfloat162float(Z_b_base[j * p.stride_C_n + ch]);
                float bbj = __bfloat162float(p.B_b[j * c_out + ch]);
                z[M + j]     = zbj + bbj;
                c_val[M + j] = __bfloat162float(C_b_base[j * p.stride_C_n + ch]);
            }
        }

        // Stable softmax over 2M positions.
        float zmax = z[0];
        #pragma unroll
        for (int j = 1; j < 2 * M; ++j) {
            zmax = fmaxf(zmax, z[j]);
        }
        // If zmax == -inf (block 0 with degenerate Z_a), avoid NaN.
        if (!isfinite(zmax)) zmax = 0.0f;

        float sum_exp = 0.0f;
        float weighted[2 * M];
        #pragma unroll
        for (int j = 0; j < 2 * M; ++j) {
            float e = __expf(z[j] - zmax);
            weighted[j] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;

        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 2 * M; ++j) {
            acc += weighted[j] * inv_sum * c_val[j];
        }

        out_base[ch] = __float2bfloat16(acc);
    }
}

// Launcher. Picks THREADS based on c_out so we have at least one
// channel per thread (and at most 8 to keep registers in check).
template<int M>
inline void launch_csa_compress_reduce(const CsaCompressReduceParams& p, cudaStream_t stream) {
    constexpr int THREADS = 256;
    dim3 grid(p.n_blocks, p.B);
    dim3 block(THREADS);
    csa_compress_reduce_kernel<M, THREADS><<<grid, block, 0, stream>>>(p);
}

}  // namespace dsv4::csa::sm100
