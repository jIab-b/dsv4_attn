#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "params.h"
#include "sm100/kernel.h"

namespace dsv4::csa {

// Pybind-facing entry. Validates, fills Params, dispatches.
//
// Inputs (all bf16, contiguous, on cuda):
//   C_a, C_b, Z_a, Z_b : [B, n, c_out]   (output of pass-1 torch matmul, sliced)
//   B_a, B_b           : [m, c_out]
//   m                  : int
//
// Returns:
//   out                : [B, n_blocks, c_out] bf16   where n_blocks = n/m
inline at::Tensor csa_compress_reduce_fwd(
    const at::Tensor& C_a,
    const at::Tensor& C_b,
    const at::Tensor& Z_a,
    const at::Tensor& Z_b,
    const at::Tensor& B_a,
    const at::Tensor& B_b,
    int64_t m
) {
    TORCH_CHECK(C_a.is_cuda() && C_b.is_cuda() && Z_a.is_cuda() && Z_b.is_cuda(),
                "all inputs must be on cuda");
    TORCH_CHECK(C_a.scalar_type() == at::kBFloat16, "C_a must be bf16");
    TORCH_CHECK(C_b.scalar_type() == at::kBFloat16, "C_b must be bf16");
    TORCH_CHECK(Z_a.scalar_type() == at::kBFloat16, "Z_a must be bf16");
    TORCH_CHECK(Z_b.scalar_type() == at::kBFloat16, "Z_b must be bf16");
    TORCH_CHECK(B_a.scalar_type() == at::kBFloat16 && B_b.scalar_type() == at::kBFloat16,
                "biases must be bf16");
    TORCH_CHECK(C_a.dim() == 3 && C_a.sizes() == C_b.sizes() &&
                C_a.sizes() == Z_a.sizes() && C_a.sizes() == Z_b.sizes(),
                "C_a/C_b/Z_a/Z_b must all be [B, n, c_out] and equal-shaped");
    TORCH_CHECK(C_a.is_contiguous() && C_b.is_contiguous() &&
                Z_a.is_contiguous() && Z_b.is_contiguous(),
                "input streams must be contiguous");
    TORCH_CHECK(B_a.dim() == 2 && B_a.size(0) == m && B_b.size(0) == m,
                "B_a/B_b must be [m, c_out]");
    TORCH_CHECK(B_a.size(1) == C_a.size(2) && B_b.size(1) == C_a.size(2),
                "B_a/B_b last dim must match c_out");

    const int64_t B = C_a.size(0);
    const int64_t n = C_a.size(1);
    const int64_t c_out = C_a.size(2);
    TORCH_CHECK(n % m == 0, "n must be divisible by m");
    const int64_t n_blocks = n / m;

    auto opts = C_a.options();
    at::Tensor out = at::empty({B, n_blocks, c_out}, opts);

    CsaCompressReduceParams p{};
    p.B           = static_cast<int>(B);
    p.n           = static_cast<int>(n);
    p.n_blocks    = static_cast<int>(n_blocks);
    p.c_out       = static_cast<int>(c_out);
    p.m           = static_cast<int>(m);
    p.C_a         = reinterpret_cast<const __nv_bfloat16*>(C_a.data_ptr());
    p.C_b         = reinterpret_cast<const __nv_bfloat16*>(C_b.data_ptr());
    p.Z_a         = reinterpret_cast<const __nv_bfloat16*>(Z_a.data_ptr());
    p.Z_b         = reinterpret_cast<const __nv_bfloat16*>(Z_b.data_ptr());
    p.B_a         = reinterpret_cast<const __nv_bfloat16*>(B_a.data_ptr());
    p.B_b         = reinterpret_cast<const __nv_bfloat16*>(B_b.data_ptr());
    p.out         = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.stride_C_b  = static_cast<int>(n * c_out);
    p.stride_C_n  = static_cast<int>(c_out);
    p.stride_out_b   = static_cast<int>(n_blocks * c_out);
    p.stride_out_blk = static_cast<int>(c_out);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Runtime → constexpr dispatch on M. V4-Flash and V4-Pro both fix m=4
    // (paper §4.2.1); add new cases here if a future variant grows m.
    if (m == 4) {
        sm100::launch_csa_compress_reduce<4>(p, stream);
    } else {
        TORCH_CHECK(false, "[dsv4-csa] unsupported m=", m, " (only m=4 instantiated)");
    }
    return out;
}

}  // namespace dsv4::csa
