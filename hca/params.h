#pragma once

#include <cuda_bf16.h>

namespace dsv4::hca {

// Pass-2 kernel params: softmax-gated reduce over the pre-projected
// (C, Z) streams produced by an upstream torch matmul.
struct HcaCompressReduceParams {
    int B;            // batch
    int n;            // sequence length, must be divisible by m_prime
    int n_blocks;     // = n / m_prime
    int c_out;        // = c (= 512 in V4)
    int m_prime;      // compression rate (=128 for V4)

    // Inputs: each [B, n, c_out] bf16, contiguous.
    const __nv_bfloat16* __restrict__ C;     // H @ W_KV
    const __nv_bfloat16* __restrict__ Z;     // H @ W_Z

    // Per-position bias [m_prime, c_out] bf16.
    const __nv_bfloat16* __restrict__ bias;

    __nv_bfloat16* __restrict__ out;

    int stride_C_b;
    int stride_C_n;
    int stride_out_b;
    int stride_out_blk;
};

}  
