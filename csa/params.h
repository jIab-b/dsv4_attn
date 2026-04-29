#pragma once

#include <cuda_bf16.h>

namespace dsv4::csa {

// Pass-2 kernel params: softmax-gated reduce over the pre-projected
// (C_a, C_b, Z_a, Z_b) streams produced by an upstream torch matmul.
//
// All tensor strides are in elements (not bytes). Layout assumed
// row-major-contiguous unless overridden.
struct CsaCompressReduceParams {
    int B;            // batch (typically 1)
    int n;            // sequence length, must be divisible by m
    int n_blocks;     // = n / m
    int c_out;        // output channels (512, 128, or 640 fused)
    int m;            // compression rate (=4 for V4)

    // Inputs: each is [B, n, c_out] bf16, contiguous over c_out.
    // For unfused / fused alike, the upstream caller hands us the four
    // pre-projected streams separately.
    const __nv_bfloat16* __restrict__ C_a;   // H @ W_a_KV
    const __nv_bfloat16* __restrict__ C_b;   // H @ W_b_KV
    const __nv_bfloat16* __restrict__ Z_a;   // H @ W_a_Z
    const __nv_bfloat16* __restrict__ Z_b;   // H @ W_b_Z

    // Per-position biases: each [m, c_out] bf16.
    const __nv_bfloat16* __restrict__ B_a;
    const __nv_bfloat16* __restrict__ B_b;

    // Output: [B, n_blocks, c_out] bf16.
    __nv_bfloat16* __restrict__ out;

    // Element strides (computed by caller).
    int stride_C_b;       // B-stride for C_a / C_b / Z_a / Z_b (= n * c_out)
    int stride_C_n;       // n-stride                            (= c_out)
    int stride_out_b;     // B-stride for out                    (= n_blocks * c_out)
    int stride_out_blk;   // block-stride for out                (= c_out)
};

}  // namespace dsv4::csa
