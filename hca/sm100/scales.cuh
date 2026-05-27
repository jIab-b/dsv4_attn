///*****************************************************************************
///*** scale-factor staging (smem → tmem)
///***
///*** Glue between the producer warps (which TMA scale bytes into smem along
///*** with the latent / rope data) and the MMA issue threads (which need the
///*** scales resident in TMEM at the SCORE_SCALE_{A,B} / VALUE_SCALE_{A,B}
///*** addresses before the corresponding tcgen05.mma fires).
///***
///*** All raw asm lives in helpers.h. Functions here only chain
///*** tcgen05_cp_2sm_32x128b_warpx4 calls plus the necessary smem descriptors.
///***
///*** Each tcgen05.cp.32x128b.warpx4 moves 32 lanes × 128 bits = 512 B from
///*** smem to TMEM, multicast across all 4 warps' lane partitions — this is
///*** the per-PTX-spec "duplicate to all 32-lane partitions" requirement for
///*** mxf8 scale factors (PTX 9.7.16.10.7).
///***
///*** Issue convention: a single elected thread of the issuing warp drives the
///*** cp calls. The destination TMEM addresses are warp-broadcast by the cp
///*** opcode itself (warpx4 multicast), so no per-warp partition arithmetic is
///*** needed here.
///***
///*** Included from kernel.cu inside `namespace dsv4::hca::sm100` — matching
///*** the convention of the other component .cuh files.
///*****************************************************************************

// Bytes one .32x128b.warpx4 cp moves from smem to TMEM (lanes × bits / 8).
constexpr int SF_CP_BYTES = 32 * 128 / 8;        // 512

// ---- score MMA scales --------------------------------------------------------

// K-side score scales (SFA): K nope per-64 e8m0 scales for one buf,
// TILE_KV rows × SCALES_PER_TOKEN bytes = 128 × 7 = 896 B per CTA per tile.
// Two cp calls cover the source contiguously; the surplus past 896 B reads
// pad bytes (TMA box padded the smem region to the next 16 B alignment).
__device__ __forceinline__
void stage_score_kv_scales(const Smem& smem, int buf) {
    if (!elect_one_sync()) return;

    const auto*  src = reinterpret_cast<const char*>(&smem.u.kv.scales[buf][0]);
    const uint32_t taddr = tmem_addr(smem.tmem_start_addr, score_sf_a_col(buf));
    constexpr int N_CP = (TILE_KV * SCALES_PER_TOKEN + SF_CP_BYTES - 1) / SF_CP_BYTES;  // 2

    #pragma unroll
    for (int i = 0; i < N_CP; ++i) {
        const uint64_t sdesc = make_kmajor_smem_desc_nosw(
            src + i * SF_CP_BYTES, /*row_stride_bytes=*/SCALES_PER_TOKEN);
        tcgen05_cp_2sm_32x128b_warpx4(taddr + i * SF_CP_BYTES, sdesc);
    }
}

// Q-side score scales (SFB): staged once at prologue. Q nope scales live in
// the o_buf overlay after query_load, packed [B_H, D_NOPE/QUANT_TILE * 2] e8m0
// (= 64 × 14 = 896 B). Same shape as SFA so the same 2-cp recipe applies.
__device__ __forceinline__
void stage_score_q_scales(const Smem& smem) {
    if (!elect_one_sync()) return;

    const auto*  src = reinterpret_cast<const char*>(&smem.u.qo.o_buf[0])
                       + B_H * D_NOPE;   // post-Q-fp8 region
    const uint32_t taddr = tmem_addr(smem.tmem_start_addr, score_sf_b_col());
    constexpr int N_CP = (B_H * (D_NOPE / QUANT_TILE) * 2 + SF_CP_BYTES - 1) / SF_CP_BYTES;  // 2

    #pragma unroll
    for (int i = 0; i < N_CP; ++i) {
        const uint64_t sdesc = make_kmajor_smem_desc_nosw(
            src + i * SF_CP_BYTES, /*row_stride_bytes=*/(D_NOPE / QUANT_TILE) * 2);
        tcgen05_cp_2sm_32x128b_warpx4(taddr + i * SF_CP_BYTES, sdesc);
    }
}

// ---- value MMA scales --------------------------------------------------------

// V-side value scales (SFA): V is the same fp8 latent as K nope, so its e8m0
// per-64 scales come from the same smem.u.kv.scales[buf] region. The value MMA
// reads V transposed but the scale storage is identical — same 896 B layout.
__device__ __forceinline__
void stage_value_v_scales(const Smem& smem, int buf) {
    if (!elect_one_sync()) return;

    const auto*  src = reinterpret_cast<const char*>(&smem.u.kv.scales[buf][0]);
    const uint32_t taddr = tmem_addr(smem.tmem_start_addr, value_sf_a_col(buf));
    constexpr int N_CP = (TILE_KV * SCALES_PER_TOKEN + SF_CP_BYTES - 1) / SF_CP_BYTES;

    #pragma unroll
    for (int i = 0; i < N_CP; ++i) {
        const uint64_t sdesc = make_kmajor_smem_desc_nosw(
            src + i * SF_CP_BYTES, /*row_stride_bytes=*/SCALES_PER_TOKEN);
        tcgen05_cp_2sm_32x128b_warpx4(taddr + i * SF_CP_BYTES, sdesc);
    }
}

// P-side value scales (SFB): one e8m0 per (buf, warp, head) = 2 × B_H = 128 B
// per buf, produced by the softmax wg during scale_and_store. Fits in a single
// 32x128b.warpx4 cp (512 B), the tail is overread of adjacent smem padding.
__device__ __forceinline__
void stage_value_p_scales(const Smem& smem, int buf) {
    if (!elect_one_sync()) return;

    const auto*  src = reinterpret_cast<const char*>(&smem.value_mma_scales[buf][0][0]);
    const uint32_t taddr = tmem_addr(smem.tmem_start_addr, value_sf_b_col(buf));
    const uint64_t sdesc = make_kmajor_smem_desc_nosw(
        src, /*row_stride_bytes=*/B_H);
    tcgen05_cp_2sm_32x128b_warpx4(taddr, sdesc);
}

// Bundles both sides of the value MMA scale staging — called once per buf,
// right before the chunk-loop in value_issue_thread.
__device__ __forceinline__
void stage_value_scales_pair(const Smem& smem, int buf) {
    stage_value_v_scales(smem, buf);
    stage_value_p_scales(smem, buf);
}
