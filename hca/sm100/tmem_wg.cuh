///*****************************************************************************
///*** softmax wg
///*****************************************************************************

struct SoftmaxPipelineState {
    int  buf              = 0;
    int  phase            = 0;
    int  last_value_buf   = 0;
    int  last_value_phase = 0;
    bool released_value   = false;
};

__device__ __forceinline__ void advance_softmax_pipeline(SoftmaxPipelineState& pipe) {
    pipe.released_value   = true;
    pipe.last_value_buf   = pipe.buf;
    pipe.last_value_phase = pipe.phase;
    pipe.buf = (pipe.buf + 1) % NUM_BUFS;
    if (pipe.buf == 0) pipe.phase ^= 1;
}

__device__ __forceinline__ void init_rolling_state(Smem& smem) {
    if (threadIdx.x < B_H) {
        smem.rolling_m[threadIdx.x] = -INFINITY;
        smem.rolling_l[threadIdx.x] = 0.0f;
    }
}

// Two warps cover the CTA's 64-token slice for one head_half (= it).
// Warps 0/1 own tok_half 0/1; warps 2/3 idle (still hit barriers downstream).
// Per active thread: 32 fp32 head_score regs, alive past dsmem into scale_and_store.
__device__ __forceinline__ void load_and_shuffle(
    const HcaParams& p, const KernelState& ks, Smem& smem,
    int tile_start, int tile_end, int it, int buf, float (&head_score)[32]
) {
    const int wg_warp = ks.warp_idx & 3;
    if (wg_warp >= 2) return;

    const int tok_half   = wg_warp & 1;
    const int lane_base  = it * 64 + tok_half * 32;
    const int local_tok  = tok_half * 32 + ks.lane;
    const int global_tok = tile_start + ks.cta_rank_in_pair * 64 + local_tok;
    const bool valid     = global_tok < tile_end;

    tcgen05_ld_32x32b_x32(
        tmem_addr(smem.tmem_start_addr, p_col(buf), lane_base),
        head_score);

    for (int h = 0; h < 32; ++h)
        head_score[h] = valid ? head_score[h] * p.sm_scale_log2 : -INFINITY;

    // Per-head warp reduce. m and l are scalar — only one head's worth is
    // live at any moment, so total reg pressure stays at the 32 head_score regs.
    for (int h = 0; h < 32; ++h) {
        float m = head_score[h];
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));

        float l = exp2f(head_score[h] - m);
        for (int off = 16; off > 0; off >>= 1)
            l += __shfl_xor_sync(0xffffffff, l, off);

        if (ks.lane == 0) {
            smem.partial_m[wg_warp][h] = m;
            smem.partial_l[wg_warp][h] = l;
        }
    }
}

// Cluster-wide partials exchange + 4-way combine for this iter's 32 heads.
// In:  partial_{m,l}[0..1][h] populated by load_and_shuffle.
// Out: curr_m[it*32+h] := tile_m, curr_l[it*32+h] := tile_l (warp 0 lanes).
__device__ __forceinline__ void dsmem_reduce(
    const KernelState& ks, Smem& smem, int it
) {
    const int wg_warp = ks.warp_idx & 3;

    sync_softmax_wg(smem);

    if (wg_warp == 0 && ks.lane == 0) {
        auto* peer_bar = peer_ptr(&smem.mbar_peer_partials);
        // 2 warps × {m, l} × 32 heads × 4 B = 512 B
        dsmem_expect_tx(peer_bar, 2 * 2 * 32 * sizeof(float));
        dsmem_copy_async(peer_ptr(&smem.peer_partial_m[0][0]),
                         &smem.partial_m[0][0],
                         2 * 32 * sizeof(float), peer_bar);
        dsmem_copy_async(peer_ptr(&smem.peer_partial_l[0][0]),
                         &smem.partial_l[0][0],
                         2 * 32 * sizeof(float), peer_bar);
    }

    dsmem_wait(smem.mbar_peer_partials);

    if (wg_warp == 0) {
        const int h = ks.lane;
        const float m0 = smem.partial_m[0][h], l0 = smem.partial_l[0][h];
        const float m1 = smem.partial_m[1][h], l1 = smem.partial_l[1][h];
        const float m2 = smem.peer_partial_m[0][h], l2 = smem.peer_partial_l[0][h];
        const float m3 = smem.peer_partial_m[1][h], l3 = smem.peer_partial_l[1][h];

        const float tile_m = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
        const float tile_l =
            l0 * exp2f(m0 - tile_m) +
            l1 * exp2f(m1 - tile_m) +
            l2 * exp2f(m2 - tile_m) +
            l3 * exp2f(m3 - tile_m);

        smem.curr_m[it * 32 + h] = tile_m;
        smem.curr_l[it * 32 + h] = tile_l;
    }

    sync_softmax_wg(smem);
}

// Online-softmax rolling update for this iter's 32 heads. Repurposes curr_*:
//   in : curr_m = tile_m, curr_l = tile_l
//   out: rolling_{m,l} folded; curr_m := new_m, curr_l := alpha
//        (new_m → scale_and_store, alpha → value_release O rescale)
__device__ __forceinline__ void update_rolling(Smem& smem, int it) {
    const int wg_warp = (threadIdx.x / 32) & 3;
    if (wg_warp == 0) {
        const int lane = threadIdx.x & 31;
        const int head = it * 32 + lane;

        const float tile_m = smem.curr_m[head];
        const float tile_l = smem.curr_l[head];
        const float old_m  = smem.rolling_m[head];
        const float old_l  = smem.rolling_l[head];

        const float new_m = fmaxf(old_m, tile_m);
        const float alpha = (old_m == -INFINITY) ? 0.0f : exp2f(old_m  - new_m);
        const float beta  = exp2f(tile_m - new_m);

        smem.rolling_m[head] = new_m;
        smem.rolling_l[head] = old_l * alpha + tile_l * beta;
        smem.curr_m[head]    = new_m;
        smem.curr_l[head]    = alpha;
    }
    sync_softmax_wg(smem);
}

// Final exp/cvt/store using the live head_score regs. Layout written:
// softmax[buf] is K-major [head, token_inner] with TILE_KV stride.
// Each active thread writes 32 fp8 bytes at strided head positions for one token.
__device__ __forceinline__ void scale_and_store(
    const KernelState& ks, Smem& smem,
    int buf, int it, float (&head_score)[32]
) {
    const int wg_warp = ks.warp_idx & 3;
    if (wg_warp >= 2) return;

    const int tok_half  = wg_warp & 1;
    const int local_tok = tok_half * 32 + ks.lane;
    const int head_base = it * 32;
    const int tok_base  = ks.cta_rank_in_pair * 64 + local_tok;

    auto* dst = reinterpret_cast<uint8_t*>(&smem.u.kv.softmax[buf][0]);

    for (int h = 0; h < 32; h += 2) {
        const int hi0 = head_base + h;
        const int hi1 = head_base + h + 1;

        // Per-(warp, head) MXFP8 e8m0 from already-reduced partial_m.
        // All lanes compute the same value (smem broadcast read); lane 0 publishes.
        const float bm0 = exp2f(smem.partial_m[wg_warp][h]     - smem.curr_m[hi0]);
        const float bm1 = exp2f(smem.partial_m[wg_warp][h + 1] - smem.curr_m[hi1]);
        const uint8_t e8_0 = f32_to_e8m0_rp(bm0 * (1.f / 448.f));
        const uint8_t e8_1 = f32_to_e8m0_rp(bm1 * (1.f / 448.f));
        if (ks.lane == 0) {
            smem.value_mma_scales[buf][wg_warp][hi0] = e8_0;
            smem.value_mma_scales[buf][wg_warp][hi1] = e8_1;
        }
        const float inv_0 = 1.f / e8m0_to_f32(e8_0);
        const float inv_1 = 1.f / e8m0_to_f32(e8_1);

        const float p0 = exp2f(head_score[h]     - smem.curr_m[hi0]) * inv_0;
        const float p1 = exp2f(head_score[h + 1] - smem.curr_m[hi1]) * inv_1;
        uint16_t pair;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(pair) : "f"(p1), "f"(p0));
        dst[hi0 * TILE_KV + tok_base] = uint8_t(pair & 0xff);
        dst[hi1 * TILE_KV + tok_base] = uint8_t((pair >> 8) & 0xff);
    }
}

__device__ __forceinline__ void consume_score(
    const HcaParams& p, const KernelState& ks, Smem& smem,
    int tile_start, int tile_end,
    SoftmaxPipelineState& pipe
) {
    mbar_wait(smem.mbar_qk_done[pipe.buf], pipe.phase);
    tcgen05_fence_after_mma();

    for (int it = 0; it < 2; ++it) {
        float head_score[32];
        load_and_shuffle(p, ks, smem, tile_start, tile_end, it, pipe.buf, head_score);
        dsmem_reduce(ks, smem, it);
        update_rolling(smem, it);
        scale_and_store(ks, smem, pipe.buf, it, head_score);
    }

    // All 128 wg threads must finish before either downstream-facing arrive:
    //   p_consumed → score wg may reuse P[buf] (TMEM)
    //   alpha_ready → epi wg may read curr_l (smem) for chunk rescale
    // mbar counts are 1 (single-thread arrive); the wg-wide sync is the
    // memory-ordering point.
    sync_softmax_wg(smem);
    if ((ks.warp_idx & 3) == 0 && ks.lane == 0) {
        mbar_arrive(smem.mbar_p_consumed [pipe.buf]);
        mbar_arrive(smem.mbar_alpha_ready[pipe.buf]);
    }
}

__device__ __inline__ void softmax_warpgroup(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    init_rolling_state(smem);
    SoftmaxPipelineState pipe{};

    auto run = [&](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            consume_score(p, ks, smem, r, r_end, pipe);
            advance_softmax_pipeline(pipe);
        }
    };
    run(ks.compressed_start, ks.compressed_end);
    run(ks.swa_start,        ks.swa_end);
}

///*****************************************************************************
///*** end softmax wg
///*****************************************************************************
