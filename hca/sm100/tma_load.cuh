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
/// @post Caller must follow with `__syncthreads()`. No barrier inside - only
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

    constexpr int REPLIC = QUANT_TILE / 32;                     // 2 (post-flip)
    constexpr int SPR    = (D_NOPE / QUANT_TILE) * REPLIC;      // 14 (post-flip)
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
