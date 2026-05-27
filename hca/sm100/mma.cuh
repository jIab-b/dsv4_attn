///*****************************************************************************
///*** mma issue loop
///***
///*** Two single-elected threads (lane 0 = score, lane 1 = value) drive every
///*** tcgen05.mma issue for the warpgroup. score_issue_thread folds nope (mxf8)
///*** and rope (bf16) into a single P tile per buf; value_issue_thread walks
///*** the chunked epi handshakes per buf.
///***
///*** All raw PTX (cp / mma / commit / fence) lives in helpers.h. Scale-factor
///*** staging is delegated to scales.cuh.
///*****************************************************************************

__device__ __forceinline__
void advance_issue_ring(int& buf, int& phase) {
    buf = (buf + 1) % NUM_BUFS;
    if (buf == 0) phase ^= 1;
}

// ---- score: nope (14 mxf8 K-blocks) + rope (4 bf16 K-blocks), accumulating
//      into P[buf]. K-side score scales are staged per tile per buf; Q-side
//      score scales are staged once at warpgroup entry (see mma_warp below).
__device__ __forceinline__
void issue_score_tile(const Smem& smem, int buf) {
    #pragma unroll
    for (int kb = 0; kb < SCORE_K_BLOCKS; ++kb) {
        score_mma(smem, buf, kb, /*accumulate=*/kb != 0);
    }
    #pragma unroll
    for (int kb = 0; kb < ROPE_K_BLOCKS; ++kb) {
        score_mma_rope(smem, buf, kb, /*accumulate=*/true);
    }
}

// ---- value: VALUE_TOKEN_BLOCKS mxf8 K-blocks for one chunk, accumulating
//      into O[chunk]. `value_started[chunk]` flips to true after the chunk's
//      first MMA on this tile so subsequent token-blocks accumulate; across
//      tiles the running O sum lives in the same O[chunk] cols.
__device__ __forceinline__
void issue_value_chunk(
    const Smem& smem, int buf, int chunk, bool (&value_started)[VALUE_DIM_BLOCKS]
) {
    #pragma unroll
    for (int tk = 0; tk < VALUE_TOKEN_BLOCKS; ++tk) {
        value_mma(smem, buf, chunk, tk,
                  /*accumulate=*/value_started[chunk] || tk != 0);
    }
    value_started[chunk] = true;
}

__device__ __inline__ void score_issue_thread(
    const HcaParams&, const KernelState& ks, Smem& smem
) {
    int buf   = 0;
    int phase = 0;
    int iter  = 0;

    auto run_ring = [&](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            // KV (nope + scales + rope) landed for this buf.
            mbar_wait(smem.mbar_kv_ready[buf], phase);
            // P[buf] is double-buffered, but SCORE_SCALE_A[buf] sits on the
            // same ring. After NUM_BUFS in-flight tiles, the softmax wg must
            // have drained P[buf] before we issue the next score into it.
            if (iter >= NUM_BUFS)
                mbar_wait(smem.mbar_p_consumed[buf], phase ^ 1);
            tcgen05_fence_after_mma();

            stage_score_kv_scales(smem, buf);
            issue_score_tile(smem, buf);
            tcgen05_commit(smem.mbar_qk_done[buf]);

            ++iter;
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
            // Value SF are reused across both chunks of the tile — stage once
            // per buf, outside the chunk loop. V scales = K scales (same fp8
            // latent); P scales come from softmax via smem.value_mma_scales.
            stage_value_scales_pair(smem, buf);

            #pragma unroll
            for (int c = 0; c < VALUE_DIM_BLOCKS; ++c) {
                mbar_wait(smem.mbar_so_ready_chunk[c][buf], phase);
                tcgen05_fence_after_mma();
                issue_value_chunk(smem, buf, c, value_started);
                tcgen05_commit(smem.mbar_sv_done_chunk[c][buf]);
            }

            advance_issue_ring(buf, phase);
        }
    };

    run_ring(ks.compressed_start, ks.compressed_end);
    run_ring(ks.swa_start,        ks.swa_end);
}

__device__ __inline__ void mma_warp(
    const HcaParams& p, const KernelState& ks, Smem& smem
) {
    // Q-side score scales: Q is loaded once at prologue (query_load before the
    // warpgroup branches). Stage SFB into TMEM here once, reused for every
    // score MMA across the KV loop.
    stage_score_q_scales(smem);
    __syncwarp();

    if (ks.lane == 0) {
        score_issue_thread(p, ks, smem);
    } else if (ks.lane == 1) {
        value_issue_thread(p, ks, smem);
    }
}

///*****************************************************************************
///*** end mma issue loop
///*****************************************************************************
