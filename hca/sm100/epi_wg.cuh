///*****************************************************************************
///*** post-value epilogue warpgroup
///***
///*** Staggers the O-rescale across the value MMA's per-chunk completion.
///*** Each of N_CHUNKS chunks (= VALUE_DIM_BLOCKS per CTA) has its own
///*** sv_done ↔ so_ready handshake; alpha is published once per tile via
///*** mbar_alpha_ready. The first tile bootstraps (no rescale, just unblock
///*** the chain).
///***
///*** Within a chunk we do the two 32-col half-loads sequentially, reusing
///*** the same 32-register scratch — staggering lets us avoid holding the
///*** full 64-reg chunk live.
///***
///*** Per chunk per tile:
///***   wait mbar_sv_done_chunk[chunk][prev_buf]    ← prior value MMA done
///***   wait mbar_alpha_ready[buf]                  ← softmax wg published α
///***   for half ∈ {0, 1}:
///***     ld 32 fp32 → α-rescale in regs → st 32 fp32   (in place)
///***   arrive mbar_so_ready_chunk[chunk][buf]      ← value MMA can fire
///***
///*** mbar_so_ready_chunk and mbar_sv_done_chunk are cluster-arrived so the
///*** cta_group::2 MMA blocks until both CTAs finish their chunk rescale.
///*** mbar_alpha_ready is per-CTA local (each CTA's softmax wg writes its own).
///*****************************************************************************

constexpr int N_CHUNKS   = VALUE_DIM_BLOCKS;          // = 2 per CTA
constexpr int CHUNK_COLS = VALUE_N;                   // = 64 TMEM cols / chunk
constexpr int HALF_COLS  = CHUNK_COLS / 2;            // = 32 TMEM cols / sub-load

struct EpiloguePipelineState {
    int  buf              = 0;
    int  phase            = 0;
    int  last_value_buf   = 0;
    int  last_value_phase = 0;
    int  alpha_phase      = 0;
    bool released_value   = false;
};

__device__ __forceinline__
void advance_epi_pipeline(EpiloguePipelineState& pipe) {
    pipe.released_value   = true;
    pipe.last_value_buf   = pipe.buf;
    pipe.last_value_phase = pipe.phase;
    pipe.buf              = (pipe.buf + 1) % NUM_BUFS;
    if (pipe.buf == 0) pipe.phase ^= 1;
    pipe.alpha_phase ^= 1;
}

__device__ __forceinline__
int chunk_tmem_col(int chunk) {
    return tmem_cols::O + chunk * CHUNK_COLS;
}

// Rescale one 32-col half of a chunk in place. Reuses 32 fp32 regs.
__device__ __forceinline__
void rescale_half(Smem& smem, int tmem_col, float alpha) {
    float o[32];
    tcgen05_ld_32x32b_x32(tmem_addr(smem.tmem_start_addr, tmem_col), o);
    #pragma unroll
    for (int i = 0; i < 32; ++i) o[i] *= alpha;
    tcgen05_st_32x32b_x32(tmem_addr(smem.tmem_start_addr, tmem_col), o);
}

// One chunk: 2 mbar waits, 2 sequential 32-reg half-rescales, 1 cluster arrive.
__device__ __forceinline__
void rescale_chunk(
    Smem& smem,
    int chunk,
    int buf, int prev_buf,
    int prev_phase, int alpha_phase
) {
    mbar_wait(smem.mbar_sv_done_chunk[chunk][prev_buf], prev_phase);
    tcgen05_fence_after_mma();
    mbar_wait(smem.mbar_alpha_ready[buf], alpha_phase);

    const int   my_head = threadIdx.x & (B_H - 1);
    const float alpha   = smem.curr_l[my_head];
    const int   base    = chunk_tmem_col(chunk);

    rescale_half(smem, base,             alpha);
    rescale_half(smem, base + HALF_COLS, alpha);

    mbar_arrive(smem.mbar_so_ready_chunk[chunk][buf]);
}

// Final drain: wait all chunks of last tile, normalize O by rolling_l,
// write partial_O + partial_lse to gmem. 32 regs live at a time.
__device__ __forceinline__
void epi_drain(
    const HcaParams& p, const KernelState& ks, Smem& smem,
    const EpiloguePipelineState& pipe
) {
    if (!pipe.released_value) return;

    #pragma unroll
    for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
        mbar_wait(smem.mbar_sv_done_chunk[chunk][pipe.last_value_buf],
                  pipe.last_value_phase);
    }
    tcgen05_fence_after_mma();

    if (ks.partial_O == nullptr || ks.partial_lse == nullptr) return;

    const int   idx     = threadIdx.x;
    const int   row     = idx & (B_H - 1);          // head
    const int   half    = idx / B_H;                // value_dim half within chunk
    const float li      = smem.rolling_l[row];
    const float o_scale = (li == 0.0f) ? 0.0f : __fdividef(1.0f, li);

    if (idx < B_H) {
        ks.partial_lse[row] = (li == 0.0f)
            ? -INFINITY
            : smem.rolling_m[row] + log2f(li);
    }

    float* out = ks.partial_O + int64_t(row) * p.stride_partial_O_h;
    #pragma unroll
    for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
        const int base    = chunk_tmem_col(chunk);
        const int out_col = ks.partial_O_out_col[chunk][half];   // layout-dependent

        float o[32];
        tcgen05_ld_32x32b_x32(tmem_addr(smem.tmem_start_addr, base), o);
        #pragma unroll
        for (int i = 0; i < 32; ++i) out[out_col + i] = o[i] * o_scale;

        tcgen05_ld_32x32b_x32(tmem_addr(smem.tmem_start_addr, base + HALF_COLS), o);
        #pragma unroll
        for (int i = 0; i < 32; ++i) out[out_col + HALF_COLS + i] = o[i] * o_scale;
    }
}

__device__ __inline__
void epi_wg(const HcaParams& p, const KernelState& ks, Smem& smem) {
    EpiloguePipelineState pipe{};

    auto run = [&](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r += TILE_KV) {
            #pragma unroll
            for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
                if (!pipe.released_value) {
                    mbar_arrive(smem.mbar_so_ready_chunk[chunk][pipe.buf]);
                } else {
                    rescale_chunk(smem,
                                  chunk,
                                  pipe.buf, pipe.last_value_buf,
                                  pipe.last_value_phase, pipe.alpha_phase);
                }
            }
            advance_epi_pipeline(pipe);
        }
    };

    run(ks.compressed_start, ks.compressed_end);
    run(ks.swa_start,        ks.swa_end);

    epi_drain(p, ks, smem, pipe);
}

///*****************************************************************************
///*** end post-value epilogue warpgroup
///*****************************************************************************
