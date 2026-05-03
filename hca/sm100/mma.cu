///*****************************************************************************
///*** mma issue loop
///*****************************************************************************

__device__ __forceinline__
void advance_issue_ring(int& buf, int& phase) {
    buf = (buf + 1) % NUM_BUFS;
    if (buf == 0) phase ^= 1;
}

__device__ __forceinline__
void tmem_store_e8m0_ones_x16(uint32_t addr) {
    constexpr uint32_t one = 0x7f7f7f7f;  // four packed e8m0(1.0) bytes
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
        "{%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};"
        :: "r"(addr), "r"(one) : "memory");
}

__device__ __forceinline__
void init_scale_tmem_ones(Smem& smem) {
    tmem_store_e8m0_ones_x16(score_scale_a_tmem(smem));
    tmem_store_e8m0_ones_x16(score_scale_b_tmem(smem));
    tmem_store_e8m0_ones_x16(value_scale_a_tmem(smem));
    tmem_store_e8m0_ones_x16(value_scale_b_tmem(smem));
}

__device__ __forceinline__
void stage_score_scales(Smem&, int, int) {
}

__device__ __forceinline__
void stage_value_scales(Smem&, int, int, int) {
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
    init_scale_tmem_ones(smem);
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
