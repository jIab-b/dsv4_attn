#pragma once

#include <torch/extension.h>

#include "params.h"
#include "sm100/kernel.h"

namespace dsv4::hca {

inline at::Tensor hca_compress_reduce_fwd(
    const at::Tensor& C,
    const at::Tensor& Z,
    const at::Tensor& bias,
    int64_t m_prime
) {
    TORCH_CHECK(C.is_cuda() && Z.is_cuda() && bias.is_cuda());
    TORCH_CHECK(C.scalar_type() == at::kBFloat16 && Z.scalar_type() == at::kBFloat16
                && bias.scalar_type() == at::kBFloat16);
    TORCH_CHECK(C.dim() == 3 && C.sizes() == Z.sizes());
    TORCH_CHECK(C.is_contiguous() && Z.is_contiguous());
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == m_prime && bias.size(1) == C.size(2));

    const int64_t Bs = C.size(0);
    const int64_t n  = C.size(1);
    const int64_t c_out = C.size(2);
    TORCH_CHECK(n % m_prime == 0);
    const int64_t n_blocks = n / m_prime;

    at::Tensor out = at::empty({Bs, n_blocks, c_out}, C.options());

    HcaCompressReduceParams p{};
    p.B           = static_cast<int>(Bs);
    p.n           = static_cast<int>(n);
    p.n_blocks    = static_cast<int>(n_blocks);
    p.c_out       = static_cast<int>(c_out);
    p.m_prime     = static_cast<int>(m_prime);
    p.C           = reinterpret_cast<const __nv_bfloat16*>(C.data_ptr());
    p.Z           = reinterpret_cast<const __nv_bfloat16*>(Z.data_ptr());
    p.bias        = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
    p.out         = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.stride_C_b      = static_cast<int>(n * c_out);
    p.stride_C_n      = static_cast<int>(c_out);
    p.stride_out_b    = static_cast<int>(n_blocks * c_out);
    p.stride_out_blk  = static_cast<int>(c_out);

    TORCH_CHECK(m_prime == 128);
    constexpr int THREADS = 256;
    dim3 grid(p.n_blocks, p.B);
    if (n_blocks * Bs >= 74) {
        sm100::hca_compress_kernel<128, THREADS><<<grid, THREADS>>>(p);
    } else {
        constexpr int K_SPLITS = 8;
        dim3 grid_sk(p.n_blocks, p.B, K_SPLITS);
        sm100::hca_compress_splitk_kernel<128, THREADS, K_SPLITS><<<grid_sk, THREADS>>>(p);
    }
    return out;
}

}
