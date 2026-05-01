#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../params.h"
#include "helpers.h"

namespace dsv4::hca::sm100 {

// ============== warp role stubs (to be filled in) ==============
__device__ __inline__ void softmax_warpgroup       (const HcaParams&, Smem&) {}
__device__ __inline__ void compress_branch_warp    (const HcaParams&, Smem&) {}
__device__ __inline__ void dequant_warpgroup       (const HcaParams&, Smem&) {}


__device__ __inline__ void mma_warp(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    if (!cute::elect_one_sync()) return;

    // ---- one-time: Q smem -> TMEM ----
    mbar_wait(smem.bar_q_tma, 0);
    utccp_q_to_tmem(smem);
    mbar_arrive(smem.bar_q_utccp);

    // optional: wait for in-flight compression slot if it just published
    if (p.partial_count == 0) {
        mbar_wait(smem.bar_cprss_done, 0);
    }

    int buf   = 0;
    int phase = 0;

    auto run_ring = [&](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            mbar_wait(smem.bar_rope_ready[buf], phase);
            mbar_wait(smem.bar_raw_ready [buf], phase);

            tcgen05_mma_qk_rope(smem, buf);                // rope -> P (init)
            #pragma unroll
            for (int kb = 0; kb < 4; ++kb) {               // nope: 4× K=128 chunks
                tcgen05_mma_qk_nope(smem, buf, kb);

            }

            mbar_arrive(smem.bar_qk_done[buf]);

            mbar_wait(smem.bar_so_ready[buf], phase);
            // ts variant: A = S (tmem), B = V (smem); accumulates into O (tmem)
            tcgen05_mma_pv_ts(smem, buf, /*v_smem_off=*/  0, /*o_col_off=*/  0);
            tcgen05_mma_pv_ts(smem, buf, /*v_smem_off=*/256, /*o_col_off=*/128);

            mbar_arrive(smem.bar_sv_done[buf]);

            buf = (buf + 1) % NUM_BUFS;
            if (buf == 0) phase ^= 1;
        }
    };

    run_ring(ks.compressed_start, ks.compressed_end);
    run_ring(ks.swa_start, ks.swa_end);
}




template<int CPRSS_NUM>
__device__ __inline__ void hca_compress(
    const HcaParams& p,
    Smem& smem,
    int b,
    int channel_s
) {
    const __nv_bfloat16* base_C    = p.C    + b * p.stride_C_b + channel_s;
    const __nv_bfloat16* base_Z    = p.Z    + b * p.stride_C_b + channel_s;
    const __nv_bfloat16* base_bias = p.bias                    + channel_s;
          __nv_fp8_e4m3* base_out  = p.Kc   + b * p.stride_Kc_b
                                            + p.M_cur * p.stride_Kc_n
                                            + channel_s;
    // TODO: emit per-block e8m0 into p.Kc_scales

    for (int c = threadIdx.x; c < CPRSS_NUM; c += blockDim.x) {
        float m   = -INFINITY;
        float l   = 0.f;
        float acc = 0.f;

        for (int i = 0; i < M_PRIME; ++i) {
            const int row = i * p.stride_C_n;
            float zi = (float)ldg_bf16(base_Z    + row + c)
                     + (float)ldg_bf16(base_bias + row + c);
            float ci = (float)ldg_bf16(base_C    + row + c);

            float m_new = fmaxf(m, zi);
            float alpha = __expf(m  - m_new);
            float beta  = __expf(zi - m_new);
            l   = l   * alpha + beta;
            acc = acc * alpha + beta * ci;
            m   = m_new;
        }

        base_out[c] = __nv_fp8_e4m3(acc / l);
    }
}


template<int M_PRIME_T, int THREADS>
__global__ void hca_compress_kernel(HcaParams p) {
    Smem& smem = shared_state();
}


__device__ __inline__ void nope_prod_warp(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    if (!cute::elect_one_sync()) return;

    int buf   = 0;
    int phase = 0;
    int iter  = 0;

    auto run_ring = [&](const CUtensorMap* tma_K,
                        const CUtensorMap* tma_S,
                        int r_start, int r_end)
    {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            if (iter >= NUM_BUFS)
                mbar_wait(smem.bar_sv_done[buf], phase ^ 1);

            int tx = TILE_KV * D_NOPE
                   + TILE_KV * SCALES_PER_TOKEN;
            mbar_expect(smem.bar_raw_ready[buf], tx);

            tma_load_3d(tma_K, ks.batch_idx, r, 0,
                        &smem.u.kv.latent[buf], smem.bar_raw_ready[buf]);
            tma_load_3d(tma_S, ks.batch_idx, r, 0,
                        &smem.u.kv.scales[buf], smem.bar_raw_ready[buf]);

            ++iter;
            buf = (buf + 1) % NUM_BUFS;
            if (buf == 0) phase ^= 1;
        }
    };
    run_ring(&p.tma_Kc,   &p.tma_Kc_scales,   ks.compressed_start, ks.compressed_end);
    run_ring(&p.tma_Kswa, &p.tma_Kswa_scales, ks.swa_start,        ks.swa_end);
}


__device__ __inline__ void rope_prod_warp(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    if (!cute::elect_one_sync()) return;

    int buf   = 0;
    int phase = 0;
    int iter  = 0;

    auto run_ring = [&](const CUtensorMap* tma_R,
                        int r_start, int r_end)
    {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            if (iter >= NUM_BUFS)
                mbar_wait(smem.bar_qk_done[buf], phase ^ 1);

            int tx = TILE_KV * D_ROPE * sizeof(__nv_bfloat16);
            mbar_expect(smem.bar_rope_ready[buf], tx);

            tma_load_3d(tma_R, ks.batch_idx, r, 0,
                        &smem.u.kv.rope[buf], smem.bar_rope_ready[buf]);

            ++iter;
            buf = (buf + 1) % NUM_BUFS;
            if (buf == 0) phase ^= 1;
        }
    };

    run_ring(&p.tma_Kc_rope,   ks.compressed_start, ks.compressed_end);
    run_ring(&p.tma_Kswa_rope, ks.swa_start,        ks.swa_end);
}


__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
hca_decode_kernel(__grid_constant__ const HcaParams p) {
    Smem& smem = shared_state();

    prefetch_tma_descriptors(p);
    init_smem(smem);

    KernelState ks;
    init_state(ks, p);

    const int wg       = threadIdx.x / 128;
    const int warp_idx = threadIdx.x / 32;

    if (warp_idx == 0) query_tma(p, ks, smem);

    if (wg == 0) {
        softmax_warpgroup(p, smem);
    } else if (wg == 1) {
        switch (warp_idx) {
            case 4: mma_warp             (p, ks, smem); break;
            case 5: nope_prod_warp       (p, ks, smem); break;
            case 6: rope_prod_warp       (p, ks, smem); break;
            case 7: compress_branch_warp (p, ks, smem); break;
        }
    } else if (wg == 2) {
        dequant_warpgroup(p, smem);
    }
}

}
