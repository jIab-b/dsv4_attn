#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "kernel.h"
#include "helpers.h"

namespace dsv4::hca::sm100 {

///*****************************************************************************
///*** softmax / value epilogue
///*****************************************************************************

struct SoftmaxPipelineState {
    int buf = 0;
    int phase = 0;
    int last_value_buf = 0;
    int last_value_phase = 0;
    bool released_value = false;
};

struct SoftmaxTileUpdate {
    float old_m;
    float old_l;
    float new_m;
    float new_l;
    float alpha;
    bool rescale_o;
};

__device__ __forceinline__ void softmax_wg_sync(Smem& smem, int& phase) {
    mbar_arrive(smem.mbar_softmax);
    mbar_wait(smem.mbar_softmax, phase);
    phase ^= 1;
}

__device__ __forceinline__ void init_softmax_accum(Smem& smem) {
    const int row = threadIdx.x & (B_H - 1);
    smem.rolling_m[row] = -INFINITY;
    smem.rolling_l[row] = 0.0f;
    smem.curr_m[row]    = -INFINITY;
    smem.curr_l[row]    = 0.0f;
}

__device__ __forceinline__ void advance_softmax_pipeline(SoftmaxPipelineState& pipe) {
    pipe.released_value = true;
    pipe.last_value_buf = pipe.buf;
    pipe.last_value_phase = pipe.phase;
    pipe.buf = (pipe.buf + 1) % NUM_BUFS;
    if (pipe.buf == 0) pipe.phase ^= 1;
}

__device__ __forceinline__ void wait_score_mma(
    Smem& smem, const SoftmaxPipelineState& pipe
) {
    mbar_wait(smem.mbar_qk_done[pipe.buf], pipe.phase);
    tcgen05_fence_after_mma();
}

__device__ __forceinline__ void load_score_tile(
    const HcaParams& p,
    const Smem& smem,
    int tile_start,
    int tile_end,
    float (&score)[32]
) {
    const int idx       = threadIdx.x;       // wg0 only: 0..127
    const int col_base  = (idx >= B_H) ? 32 : 0;

    tcgen05_ld_32x32b_x32(
        tmem_addr(smem.tmem_start_addr, tmem_cols::P),
        score);

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int token_col = col_base + i;
        score[i] = (tile_start + token_col < tile_end)
                 ? score[i] * p.sm_scale_log2
                 : -INFINITY;
    }
}

__device__ __forceinline__ SoftmaxTileUpdate update_softmax_tile(
    Smem& smem,
    float (&score)[32],
    int& softmax_phase
) {
    const int idx = threadIdx.x;
    const int row = idx & (B_H - 1);

    float half_max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        half_max = fmaxf(half_max, score[i]);
    }

    smem.rowwise_max[idx] = half_max;
    softmax_wg_sync(smem, softmax_phase);

    const float tile_m = fmaxf(half_max, smem.rowwise_max[idx ^ B_H]);
    const float old_m  = smem.rolling_m[row];
    const float old_l  = smem.rolling_l[row];
    const bool  rescale_o = tile_m - old_m > 6.0f;
    const float new_m  = rescale_o ? fmaxf(old_m, tile_m) : old_m;
    const float alpha  = old_m == -INFINITY ? 0.0f : exp2f(old_m - new_m);

    float half_l = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        half_l += new_m == -INFINITY ? 0.0f : exp2f(score[i] - new_m);
    }

    smem.rowwise_max[idx] = half_l;
    softmax_wg_sync(smem, softmax_phase);

    const float tile_l = half_l + smem.rowwise_max[idx ^ B_H];
    const float new_l  = fmaf(old_l, alpha, tile_l);

    smem.curr_m[row] = new_m;
    smem.curr_l[row] = new_l;
    smem.rolling_m[row] = new_m;
    smem.rolling_l[row] = new_l;

    return {old_m, old_l, new_m, new_l, alpha, rescale_o};
}

__device__ __forceinline__ void store_softmax_operand_for_value(
    Smem&,
    const float (&)[32],
    const SoftmaxTileUpdate&
) {
    // TODO: convert softmax weights to fp8 S^T in smem.u.kv.softmax[buf]
    // and stage its e8m0 scales into VALUE_SCALE_B.
}

__device__ __forceinline__ void rescale_o_accum_if_needed(
    Smem&,
    const SoftmaxPipelineState& pipe,
    const SoftmaxTileUpdate& update
) {
    // FlashMLA rescales O in TMEM here only when the rolling max jumps enough.
    // This stays behind a helper because the final O layout is still being settled.
    (void)pipe;
    (void)update;
}

__device__ __forceinline__ void release_value_mma(
    Smem& smem, SoftmaxPipelineState& pipe
) {
    mbar_arrive(smem.mbar_so_ready[pipe.buf]);
    advance_softmax_pipeline(pipe);
}

__device__ __forceinline__ void consume_score_tile(
    const HcaParams& p,
    Smem& smem,
    int tile_start,
    int tile_end,
    SoftmaxPipelineState& pipe,
    int& softmax_phase
) {
    wait_score_mma(smem, pipe);

    float score[32];
    load_score_tile(p, smem, tile_start, tile_end, score);
    SoftmaxTileUpdate update = update_softmax_tile(smem, score, softmax_phase);

    store_softmax_operand_for_value(smem, score, update);
    rescale_o_accum_if_needed(smem, pipe, update);
    release_value_mma(smem, pipe);
}

__device__ __forceinline__ void value_epilogue_final(
    const HcaParams& p,
    const KernelState& ks,
    Smem& smem,
    const SoftmaxPipelineState& pipe
) {
    if (!pipe.released_value) return;

    mbar_wait(smem.mbar_sv_done[pipe.last_value_buf], pipe.last_value_phase);
    tcgen05_fence_after_mma();

    if (ks.partial_O == nullptr || ks.partial_lse == nullptr) return;

    const int idx = threadIdx.x;             // wg0 only: 0..127
    const int row = idx & (B_H - 1);
    const int half = idx / B_H;
    const float li = smem.rolling_l[row];
    const float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li);

    if (idx < B_H) {
        ks.partial_lse[row] = li == 0.0f
            ? -INFINITY
            : smem.rolling_m[row] + log2f(li);
    }

    float* out = ks.partial_O + int64_t(row) * p.stride_partial_O_h;
    #pragma unroll
    for (int chunk = 0; chunk < (D_V / 2) / 64; ++chunk) {
        const int tmem_col = tmem_cols::O + chunk * 64;
        const int out_col =
            half * (D_V / 4)
            + (chunk * 64 >= D_V / 4 ? D_V / 2 : 0)
            + (chunk * 64) % (D_V / 4);

        float o0[32];
        float o1[32];
        tcgen05_ld_32x32b_x32(
            tmem_addr(smem.tmem_start_addr, tmem_col),
            o0);
        tcgen05_ld_32x32b_x32(
            tmem_addr(smem.tmem_start_addr, tmem_col + 32),
            o1);

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            out[out_col + i] = o0[i] * o_scale;
            out[out_col + 32 + i] = o1[i] * o_scale;
        }
    }
}

__device__ __inline__ void softmax_warpgroup(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    init_softmax_accum(smem);
    int softmax_phase = 0;
    SoftmaxPipelineState pipe;

    auto run_score_tiles = [&](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            consume_score_tile(p, smem, r, r_end, pipe, softmax_phase);
        }
    };

    run_score_tiles(ks.compressed_start, ks.compressed_end);
    run_score_tiles(ks.swa_start,        ks.swa_end);
    value_epilogue_final(p, ks, smem, pipe);
}

///*****************************************************************************
///*** end softmax / value epilogue
///*****************************************************************************


///*****************************************************************************
///*** tma producers: q / nope / rope
///*****************************************************************************

/// Loads Q from gmem and quantizes Q-nope to fp8 e4m3 + per-128 e8m0 scales.
/// On exit:
///   smem.u.qo.q_sw64       : bf16 Q-rope, SW64  (untouched after TMA)
///   smem.u.qo.o_buf[fp8]   : fp8 Q-nope [B_H, D_NOPE], plain row-major
///   o_buf + B_H*D_NOPE     : e8m0 scales [B_H, 16] = 4 per row x 4 replicas
///                            (replicated to fit mxf8f6f4 scale_vec::1X / block-32)
///
/// @pre  Called by all 128 threads of wg 0 (`if (wg == 0) query_load(...)`).
/// @post Caller must follow with `__syncthreads()`. No barrier inside — only
///       wg 0 enters here, so a block-wide barrier would deadlock wg 1.
///
/// Pipeline: thread 0 issues 2x TMA (nope SW128, rope SW64) -> wg waits ->
/// warp w owns kb=w (32 lanes x 4 bf16 = one 1x128 tile) -> warp absmax ->
/// cvt.rp.satfinite.ue8m0x2.f32 (ceil-pow2) -> cvt.rn.satfinite.e4m3x2.f32.
///
/// @warning Assumes q_sw128 is plain row-major bf16. If TMA descriptor uses
///   SW128 swizzle, lane reads see byte-permuted data; absmax is permutation-
///   invariant so the scale is still correct, but fp8 output won't preserve
///   logical column order. The score B descriptor must agree with this layout.
__device__ __inline__
void query_load(const HcaParams& p, const KernelState& ks, Smem& smem) {
    // 1) TMA: bf16 Q-nope (SW128), bf16 Q-rope (SW64).
    if (ks.warp_idx == 0 && ks.lane == 0) {
        mbar_expect(smem.mbar_q_tma, B_H * (D_NOPE + D_ROPE) * sizeof(__nv_bfloat16));
        tma_load_3d(&p.tma_Q_sw128, ks.batch_idx, ks.head_half_idx, 0,
                    smem.u.qo.q_sw128, smem.mbar_q_tma);
        tma_load_3d(&p.tma_Q_sw64,  ks.batch_idx, ks.head_half_idx, 0,
                    smem.u.qo.q_sw64,  smem.mbar_q_tma);
    }
    mbar_wait(smem.mbar_q_tma, 0);

    // 2) Quant Q-nope. warp w owns kb=w; 32 lanes cover one 1x128 tile.
    auto* q_fp8    = reinterpret_cast<__nv_fp8_e4m3*>(&smem.u.qo.o_buf[0]);
    auto* q_scales = reinterpret_cast<__nv_fp8_e8m0*>(
        reinterpret_cast<char*>(&smem.u.qo.o_buf[0]) + B_H * D_NOPE);

    constexpr int REPLIC = QUANT_TILE / 32;                     // 4
    constexpr int SPR    = (D_NOPE / QUANT_TILE) * REPLIC;      // 16
    const int kb = ks.warp_idx;

    #pragma unroll 4
    for (int row = 0; row < B_H; ++row) {
        __nv_bfloat16 v[4];
        *reinterpret_cast<uint64_t*>(v) = *reinterpret_cast<const uint64_t*>(
            &smem.u.qo.q_sw128[row * D_NOPE + kb * QUANT_TILE + ks.lane * 4]);

        float mx = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) mx = fmaxf(mx, fabsf((float)v[i]));
        mx = warp_reduce_max(mx);

        uint8_t e8  = f32_to_e8m0_rp(mx * (1.f / 448.f));
        float   inv = 1.f / e8m0_to_f32(e8);

        float vf[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) vf[i] = (float)v[i] * inv;

        *reinterpret_cast<uint32_t*>(
            &q_fp8[row * D_NOPE + kb * QUANT_TILE + ks.lane * 4]) =
                f32x4_to_e4m3x4(vf);

        if (ks.lane == 0) {
            auto* sdst = reinterpret_cast<uint8_t*>(&q_scales[row * SPR + kb * REPLIC]);
            #pragma unroll
            for (int r = 0; r < REPLIC; ++r) sdst[r] = e8;
        }
    }
}

__device__ __inline__ void nope_prod_warp(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    if (!elect_one_sync()) return;

    int buf   = 0;
    int phase = 0;
    int iter  = 0;

    auto run_ring = [&](const CUtensorMap* tma_K,
                        const CUtensorMap* tma_S,
                        int r_start, int r_end)
    {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            if (iter >= NUM_BUFS)
                mbar_wait(smem.mbar_sv_done[buf], phase ^ 1);

            int tx = TILE_KV * D_NOPE
                   + TILE_KV * SCALES_PER_TOKEN;
            mbar_expect(smem.mbar_raw_ready[buf], tx);

            tma_load_3d(tma_K, ks.batch_idx, r, 0,
                        &smem.u.kv.latent[buf], smem.mbar_raw_ready[buf]);
            tma_load_3d(tma_S, ks.batch_idx, r, 0,
                        &smem.u.kv.scales[buf], smem.mbar_raw_ready[buf]);

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
    if (!elect_one_sync()) return;

    int buf   = 0;
    int phase = 0;
    int iter  = 0;

    auto run_ring = [&](const CUtensorMap* tma_R,
                        int r_start, int r_end)
    {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            if (iter >= NUM_BUFS)
                mbar_wait(smem.mbar_qk_done[buf], phase ^ 1);

            int tx = TILE_KV * D_ROPE * sizeof(__nv_bfloat16);
            mbar_expect(smem.mbar_rope_ready[buf], tx);

            tma_load_3d(tma_R, ks.batch_idx, r, 0,
                        &smem.u.kv.rope[buf], smem.mbar_rope_ready[buf]);

            ++iter;
            buf = (buf + 1) % NUM_BUFS;
            if (buf == 0) phase ^= 1;
        }
    };

    run_ring(&p.tma_Kc_rope,   ks.compressed_start, ks.compressed_end);
    run_ring(&p.tma_Kswa_rope, ks.swa_start,        ks.swa_end);
}

///*****************************************************************************
///*** end tma producers: q / nope / rope
///*****************************************************************************


///*****************************************************************************
///*** mma issue loop
///*****************************************************************************

__device__ __forceinline__
void advance_issue_ring(int& buf, int& phase) {
    buf = (buf + 1) % NUM_BUFS;
    if (buf == 0) phase ^= 1;
}

__device__ __forceinline__
void stage_score_scales(Smem&, int, int) {
    // TODO: stage K/Q e8m0 scale columns to SCORE_SCALE_A/B for this K=32 block.
}

__device__ __forceinline__
void stage_value_scales(Smem&, int, int, int) {
    // TODO: stage V/S e8m0 scale columns to VALUE_SCALE_A/B.
}

__device__ __forceinline__
void issue_score_tile(Smem& smem, int buf) {
    #pragma unroll
    for (int kb = 0; kb < SCORE_K_BLOCKS; ++kb) {
        stage_score_scales(smem, buf, kb);
        score_mma(smem, buf, kb, /*accumulate=*/kb != 0);
    }
}

__device__ __forceinline__
void issue_value_tile(
    Smem& smem,
    int buf,
    bool (&value_started)[VALUE_DIM_BLOCKS]
) {
    #pragma unroll
    for (int dim = 0; dim < VALUE_DIM_BLOCKS; ++dim) {
        #pragma unroll
        for (int tk = 0; tk < VALUE_TOKEN_BLOCKS; ++tk) {
            stage_value_scales(smem, buf, dim, tk);
            value_mma(
                smem, buf, dim, tk,
                /*accumulate=*/value_started[dim] || tk != 0);
        }
        value_started[dim] = true;
    }
}

__device__ __inline__ void score_issue_thread(
    const HcaParams&, const KernelState& ks, Smem& smem
) {
    mbar_wait(smem.mbar_q_tma, 0);
    mbar_arrive(smem.mbar_q_utccp);

    int buf   = 0;
    int phase = 0;

    auto run_ring = [&](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            mbar_wait(smem.mbar_raw_ready[buf], phase);
            tcgen05_fence_after_mma();

            issue_score_tile(smem, buf);
            tcgen05_commit(smem.mbar_qk_done[buf]);

            advance_issue_ring(buf, phase);
        }
    };

    run_ring(ks.compressed_start, ks.compressed_end);
    run_ring(ks.swa_start,        ks.swa_end);
}

__device__ __inline__ void value_issue_thread(
    const HcaParams&, const KernelState& ks, Smem& smem
) {
    int buf   = 0;
    int phase = 0;

    bool value_started[VALUE_DIM_BLOCKS] = {};

    auto run_ring = [&](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r += TILE_KV) {

            mbar_wait(smem.mbar_so_ready[buf], phase);
            tcgen05_fence_after_mma();

            issue_value_tile(smem, buf, value_started);
            tcgen05_commit(smem.mbar_sv_done[buf]);

            advance_issue_ring(buf, phase);
        }
    };

    run_ring(ks.compressed_start, ks.compressed_end);
    run_ring(ks.swa_start, ks.swa_end);
}

__device__ __inline__ void mma_warp(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    if (ks.lane == 0) {
        score_issue_thread(p, ks, smem);
    } else if (ks.lane == 1) {
        value_issue_thread(p, ks, smem);
    }
}

///*****************************************************************************
///*** end mma issue loop
///*****************************************************************************


///*****************************************************************************
///*** compressor / legacy stubs
///*****************************************************************************

__device__ __inline__ void compress_branch_warp    (const HcaParams&, const KernelState&, Smem&) {}

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
          __nv_fp8_e8m0* base_scales = p.Kc_scales + b * p.stride_Kc_scales_b
                                                  + p.M_cur * p.stride_Kc_scales_n
                                                  + channel_s / QUANT_TILE;

    constexpr int NUM_SCALES = (CPRSS_NUM + QUANT_TILE - 1) / QUANT_TILE;
    const __nv_fp8_e8m0 one_scale(1.0f);
    for (int s = threadIdx.x; s < NUM_SCALES; s += blockDim.x) {
        base_scales[s] = one_scale;
    }

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

///*****************************************************************************
///*** end compressor / legacy stubs
///*****************************************************************************


///*****************************************************************************
///*** global kernel
///*****************************************************************************

__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
hca_decode_kernel(__grid_constant__ const HcaParams p) {
    Smem& smem = shared_state();

    prefetch_tma_descriptors(p);
    init_smem(smem);

    KernelState ks;
    init_state(ks, p);

    if (ks.wg == 0) query_load(p, ks, smem);
    __syncthreads();

    if (ks.wg == 0) {
        softmax_warpgroup(p, ks, smem);
    } else if (ks.wg == 1) {
        switch (ks.warp_idx) {
            case 4: mma_warp             (p, ks, smem); break;
            case 5: nope_prod_warp       (p, ks, smem); break;
            case 6: rope_prod_warp       (p, ks, smem); break;
            case 7: compress_branch_warp (p, ks, smem); break;
        }
    }
}

///*****************************************************************************
///*** end global kernel
///*****************************************************************************


}
