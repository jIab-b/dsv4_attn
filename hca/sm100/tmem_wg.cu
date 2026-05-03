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

__device__ __forceinline__ void compute_warp_ml(
    const HcaParams& p,
    const KernelState& ks,
    Smem& smem,
    int tile_start,
    int tile_end
) {
    // wg0 only. Each warp loads one 32-token x 32-head quadrant from local
    // CTA TMEM, reduces over token lanes, and stores 32 per-head partials.
    const int wg_warp   = ks.warp_idx & 3;
    const int head_half = wg_warp >> 1;
    const int tok_half  = wg_warp & 1;

    const int lane_base  = head_half * 64 + tok_half * 32;
    const int local_tok  = tok_half * 32 + ks.lane;
    const int global_tok = tile_start + ks.cta_rank_in_pair * 64 + local_tok;
    const bool valid     = global_tok < tile_end;

    float score[32];
    tcgen05_ld_32x32b_x32(
        tmem_addr(smem.tmem_start_addr, tmem_cols::P, lane_base),
        score);

    float m[32];
    #pragma unroll
    for (int h = 0; h < 32; ++h) {
        score[h] = valid ? score[h] * p.sm_scale_log2 : -INFINITY;
        m[h] = score[h];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int h = 0; h < 32; ++h) {
            m[h] = fmaxf(m[h], __shfl_down_sync(0xffffffff, m[h], off));
        }
    }

    float l[32];
    #pragma unroll
    for (int h = 0; h < 32; ++h) {
        m[h] = __shfl_sync(0xffffffff, m[h], 0);
        l[h] = valid ? exp2f(score[h] - m[h]) : 0.0f;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int h = 0; h < 32; ++h) {
            l[h] += __shfl_down_sync(0xffffffff, l[h], off);
        }
    }

    if (ks.lane == 0) {
        #pragma unroll
        for (int h = 0; h < 32; ++h) {
            smem.partial_m[wg_warp][h] = m[h];
            smem.partial_l[wg_warp][h] = l[h];
        }
    }
}

__device__ __forceinline__ void dsmem_reduce(
    const KernelState& ks,
    Smem& smem
) {
    const int wg_warp = ks.warp_idx & 3;

    sync_softmax_wg(smem);

    if (wg_warp == 0 && ks.lane == 0) {
        auto* peer_bar = peer_ptr(&smem.mbar_peer_partials);

        dsmem_expect_tx(peer_bar, 2 * 4 * 32 * sizeof(float));

        dsmem_copy_async(
            peer_ptr(&smem.peer_partial_m[0][0]),
            &smem.partial_m[0][0],
            4 * 32 * sizeof(float),
            peer_bar);

        dsmem_copy_async(
            peer_ptr(&smem.peer_partial_l[0][0]),
            &smem.partial_l[0][0],
            4 * 32 * sizeof(float),
            peer_bar);
    }

    dsmem_wait(smem.mbar_peer_partials);

    if (wg_warp == 0) {
        int h = ks.lane;

        #pragma unroll
        for (int hb = 0; hb < 64; hb += 32) {
            int r = hb == 0 ? 0 : 2;

            float m0 = smem.partial_m[r + 0][h];
            float l0 = smem.partial_l[r + 0][h];
            float m1 = smem.partial_m[r + 1][h];
            float l1 = smem.partial_l[r + 1][h];

            float m2 = smem.peer_partial_m[r + 0][h];
            float l2 = smem.peer_partial_l[r + 0][h];
            float m3 = smem.peer_partial_m[r + 1][h];
            float l3 = smem.peer_partial_l[r + 1][h];

            float tile_m = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
            float tile_l =
                l0 * exp2f(m0 - tile_m) +
                l1 * exp2f(m1 - tile_m) +
                l2 * exp2f(m2 - tile_m) +
                l3 * exp2f(m3 - tile_m);

            int head = hb + h;
            smem.curr_m[head] = tile_m;
            smem.curr_l[head] = tile_l;
        }
    }

    sync_softmax_wg(smem);
}

__device__ __forceinline__ void tok_reduce_x32(
    const HcaParams& p,
    Smem& smem,
    int tmem_col,
    int head_base,
    int tile_start,
    int tile_end,
    int& softmax_phase
) {
    // TODO: wg0 token-reduce primitive for P[token, head]. For 2SM M=128 the
    // score tile is token-split across the CTA pair: each CTA owns local TMEM
    // for 64 tokens x 64 heads, not a duplicated 128 x 64 tile. Four warps
    // reduce this CTA's local token slice, producing local max/sum per head in
    // smem. Then exchange those partials through cluster/peer smem, combine
    // with the peer CTA to get global tile_m/tile_l per head, and normalize
    // this CTA's local P slice. Head halves come from the 2x2 TMEM lane layout,
    // not from cta_rank * 32 column offsets.
    (void)p;
    (void)smem;
    (void)tmem_col;
    (void)head_base;
    (void)tile_start;
    (void)tile_end;
    (void)softmax_phase;
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
    const KernelState& ks,
    Smem& smem,
    int tile_start,
    int tile_end,
    SoftmaxPipelineState& pipe,
    int& softmax_phase
) {
    wait_score_mma(smem, pipe);
    compute_warp_ml(p, ks, smem, tile_start, tile_end);
    softmax_wg_sync(smem, softmax_phase);
    dsmem_reduce(ks, smem);

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
            consume_score_tile(p, ks, smem, r, r_end, pipe, softmax_phase);
        }
    };

    run_score_tiles(ks.compressed_start, ks.compressed_end);
    run_score_tiles(ks.swa_start,        ks.swa_end);
    value_epilogue_final(p, ks, smem, pipe);
}

///*****************************************************************************
///*** end softmax / value epilogue
///*****************************************************************************
