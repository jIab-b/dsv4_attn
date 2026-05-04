#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cstdint>

namespace dsv4::hca {

struct HcaParams {
    // ---- shapes / counters ----
    int B;
    int n;                  // (standalone reduce) prefill seq len
    int n_blocks;           // = n / m_prime
    int c_out;              // = 512
    int m_prime;            // = 128

    int M_cur;              // current compressed length     (decode)
    int swa_len;            // current SWA fill (≤128)       (decode)
    int partial_count;      // 0..127                        (decode)

    // ---- compressor inputs (per-step latent stream + bias) ----
    const __nv_bfloat16* __restrict__ C;        // [B, m', c_out]
    const __nv_bfloat16* __restrict__ Z;        // [B, m', c_out]
    const __nv_bfloat16* __restrict__ bias;     // [m', c_out]

    // ---- compressed cache (fp8 e4m3 latent + e8m0 per-block scales + bf16 rope) ----
    __nv_fp8_e4m3* __restrict__ Kc;             // [B, M_max, 512]   fp8 e4m3
    __nv_fp8_e8m0* __restrict__ Kc_scales;      // [B, M_max, 4]     e8m0  (1 scale / 128 ch)
    __nv_bfloat16* __restrict__ Kc_rope;        // [B, M_max, 64]    bf16

    // ---- sliding-window cache (same dtype layout) ----
    __nv_fp8_e4m3* __restrict__ Kswa;           // [B, W=128, 512]
    __nv_fp8_e8m0* __restrict__ Kswa_scales;    // [B, W=128, 4]
    __nv_bfloat16* __restrict__ Kswa_rope;      // [B, W=128, 64]

    // ---- Q / O for one decode step ----
    __nv_bfloat16* __restrict__ Q;              // [B, 1, H=128, 512]
    __nv_bfloat16* __restrict__ O;              // [B, 1, H=128, 512]
    float* __restrict__ partial_O;              // [split, B, H, 512] split-K output
    float* __restrict__ partial_lse;            // [split, B, H]      split-K log2 LSE

    // ---- strides ----
    int64_t stride_C_b, stride_C_n;
    int64_t stride_Kc_b, stride_Kc_n;
    int64_t stride_Kc_scales_b, stride_Kc_scales_n;
    int64_t stride_Kc_rope_b, stride_Kc_rope_n;
    int64_t stride_Kswa_b, stride_Kswa_n;
    int64_t stride_Kswa_scales_b, stride_Kswa_scales_n;
    int64_t stride_Kswa_rope_b, stride_Kswa_rope_n;
    int64_t stride_Q_b;
    int64_t stride_O_b;
    int64_t stride_partial_O_split, stride_partial_O_b, stride_partial_O_h;
    int64_t stride_partial_lse_split, stride_partial_lse_b;

    // legacy (standalone reduce)
    __nv_bfloat16* __restrict__ out;
    int stride_out_b;
    int stride_out_blk;

    // ---- softmax / rope ----
    float  sm_scale_log2;
    float* attn_sink;
    float  rope_theta_compressed;
    float  rope_theta_swa;

    // ---- TMA descriptors (filled host-side) ----
    CUtensorMap tma_Q_sw128;
    CUtensorMap tma_Q_sw64;
    CUtensorMap tma_O;
    CUtensorMap tma_Kc;
    CUtensorMap tma_Kc_scales;
    CUtensorMap tma_Kc_rope;
    CUtensorMap tma_Kswa;
    CUtensorMap tma_Kswa_scales;
    CUtensorMap tma_Kswa_rope;
};

}
