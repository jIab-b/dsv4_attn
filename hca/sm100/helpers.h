#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cstdint>

#include "../params.h"

namespace dsv4::hca::sm100 {

// elect.sync: returns true on exactly one lane of the warp; mirrors elect_one_sync.
__device__ __forceinline__ bool elect_one_sync() {
    int pred;
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "elect.sync _|P, 0xffffffff;\n"
        "selp.b32 %0, 1, 0, P;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

// ============== compile-time dims ==============
constexpr int B_H         = 64;
constexpr int TILE_KV     = 64;
constexpr int NUM_BUFS    = 2;
constexpr int D_NOPE      = 512;
constexpr int D_ROPE      = 64;
constexpr int D_V         = 512;
constexpr int D_Q         = D_NOPE + D_ROPE;   // 576
constexpr int M_PRIME     = 128;
constexpr int W_SWA       = 128;
constexpr int QUANT_TILE  = 128;               // e8m0 group size for fp8 cache
constexpr int SCALES_PER_TOKEN = D_NOPE / QUANT_TILE;  // 4
constexpr int NUM_THREADS = 256;               // 2 warpgroups: softmax, mma+producers

// tcgen05 cta_group used kernel-wide (must be the same for every tcgen05 op)
#define TCGEN05_CTA_GROUP "::1"

// ============== Mbarrier (storage only; logic in free functions) ==============
struct Mbarrier {
    uint64_t raw;
};

__device__ __forceinline__
uint32_t smem_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void mbar_init(Mbarrier& bar, int count) {
    uint32_t addr = smem_to_uint(&bar.raw);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbar_arrive(Mbarrier& bar) {
    uint32_t addr = smem_to_uint(&bar.raw);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"(addr));
}

__device__ __forceinline__
void mbar_expect(Mbarrier& bar, int tx_bytes) {
    uint32_t addr = smem_to_uint(&bar.raw);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbar_wait(Mbarrier& bar, int phase) {
    uint32_t addr = smem_to_uint(&bar.raw);
    int done = 0;
    while (!done) {
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
            "selp.b32 %0, 1, 0, P;\n"
            "}\n"
            : "=r"(done) : "r"(addr), "r"(phase)
        );
    }
}

// ============== Smem (fp8 e4m3 latent cache + e8m0 scales + bf16 rope/dequant) ==============
struct alignas(16) Smem {
    union {
        // Q+O phase: live before mma loop and during epilogue
        struct {
            __nv_bfloat16 q_sw128[B_H * D_NOPE];                  // 64 KB  Q nope, SW128
            __nv_bfloat16 q_sw64 [B_H * D_ROPE];                  //  8 KB  Q rope, SW64
            union {
                __nv_bfloat16 o_buf      [B_H * D_V];             // 64 KB  bf16 O for TMA store
                float         o_accum_buf[B_H * D_V];
            };
        } qo;

        // KV phase: live during mma loop
        struct {
            __nv_fp8_e4m3 latent[NUM_BUFS][TILE_KV * D_NOPE];               //  64 KB  fp8 nope, K and V both
            __nv_fp8_e8m0 scales[NUM_BUFS][TILE_KV * SCALES_PER_TOKEN];     // 0.5 KB  e8m0 per-128 scales
            __nv_bfloat16 rope  [NUM_BUFS][TILE_KV * D_ROPE];               //  16 KB  bf16 rope
        } kv;
    } u;

    float         p_exchange [4][16 * TILE_KV / 4];   // 4 KB  warp coord
    float         rowwise_max[128];                   // 0.5 KB
    uint32_t      tmem_start_addr;

    // mbarriers (inline)
    Mbarrier bar_q_tma;
    Mbarrier bar_q_utccp;
    Mbarrier bar_last_store;

    Mbarrier bar_rope_ready[NUM_BUFS];
    Mbarrier bar_raw_ready [NUM_BUFS];   // fp8 latent + scales landed
    Mbarrier bar_qk_done   [NUM_BUFS];
    Mbarrier bar_so_ready  [NUM_BUFS];
    Mbarrier bar_sv_done   [NUM_BUFS];   // also serves as "latent buf free" for nope_prod

    // compressor branch (only used when partial_count == 127)
    Mbarrier bar_cprss_in;
    Mbarrier bar_cprss_done;

    // leg merge (compressed → SWA handoff of (m, l, O))
    Mbarrier bar_leg_merge;
};

// ============== smem accessor (hides extern __shared__) ==============
__device__ __forceinline__ Smem& shared_state() {
    extern __shared__ char __smem_storage[];
    return *reinterpret_cast<Smem*>(__smem_storage);
}

// ============== PTX wrappers ==============
__device__ __forceinline__
__nv_bfloat16 ldg_bf16(const __nv_bfloat16* ptr) {
    uint16_t bits;
    asm volatile("ld.global.nc.b16 %0, [%1];" : "=h"(bits) : "l"(ptr));
    return __ushort_as_bfloat16(bits);
}

__device__ __forceinline__
void prefetch_tma_descriptor(const CUtensorMap* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc));
}

// TMA 3D load with mbarrier completion (cp.async.bulk.tensor.3d.shared::cluster.global).
// Issues a single tile load; the producing thread should already have called
// `mbar_expect` on `bar` with the correct tx-byte count before issuing this.
template<typename T>
__device__ __forceinline__
void tma_load_3d(const CUtensorMap* desc,
                 int crd0, int crd1, int crd2,
                 T* dst_smem, Mbarrier& bar) {
    uint32_t s_addr = smem_to_uint(dst_smem);
    uint32_t b_addr = smem_to_uint(&bar.raw);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(s_addr), "l"(desc), "r"(b_addr),
          "r"(crd0), "r"(crd1), "r"(crd2)
        : "memory");
}

// ============== TMEM addressing ==============
// 32-bit TMEM address: [31:16] lane index, [15:0] column index.
// `base` is the per-CTA tmem allocation base (col=0, lane=0).
__device__ __forceinline__
uint32_t tmem_addr(uint32_t base, int col, int lane = 0) {
    return base + (uint32_t(lane) << 16) + uint32_t(col);
}

// ============== smem matrix descriptor (PTX Table 40) ==============
// 64-bit descriptor; encode helper: x -> (x & 0x3FFFF) >> 4 (i.e. bits [17:4]).
//   bits  0-13 : matrix start address (encoded)
//   bits 16-29 : leading-dim byte offset (encoded)
//   bits 32-45 : stride-dim byte offset (encoded)
//   bits 46-48 : const 0b001
//   bits 49-51 : matrix base offset
//   bit  52    : leading-dim stride mode (0 = relative offset)
//   bits 53-60 : reserved 0
//   bits 61-63 : swizzle mode (0=none, 1=128B/32Batom, 2=128B, 4=64B, 6=32B)
__device__ __forceinline__
uint64_t make_smem_desc(
    uint32_t smem_addr_u32,
    uint32_t leading_byte_off,
    uint32_t stride_byte_off,
    uint32_t base_offset,
    uint32_t swizzle_mode
) {
    auto enc = [](uint32_t x) { return (x & 0x3FFFFu) >> 4; };
    uint64_t d = 0;
    d |= uint64_t(enc(smem_addr_u32))    << 0;
    d |= uint64_t(enc(leading_byte_off)) << 16;
    d |= uint64_t(enc(stride_byte_off))  << 32;
    d |= uint64_t(0b001)                 << 46;
    d |= uint64_t(base_offset & 0x7)     << 49;
    // bit 52 = 0  (relative offset mode)
    d |= uint64_t(swizzle_mode & 0x7)    << 61;
    return d;
}

// ============== instruction descriptor (PTX Table 42 — kind::f16, kind::f8f6f4, kind::tf32, kind::i8) ==============
// Dense, no-negate, no-saturate, dtype = F32.
//   atype/btype encoding for kind::f16     :  F16=0, BF16=1
//   atype/btype encoding for kind::f8f6f4  :  E4M3=0, E5M2=1, E2M3=3, E3M2=4, E2M1=5
__device__ __forceinline__
uint32_t make_idesc(
    int M, int N,
    uint32_t atype, uint32_t btype,
    bool transA, bool transB
) {
    uint32_t d = 0;
    d |= uint32_t(1)               << 4;            // dtype = F32
    d |= (atype & 0x7)             << 7;
    d |= (btype & 0x7)             << 10;
    d |= uint32_t(transA ? 1 : 0)  << 15;
    d |= uint32_t(transB ? 1 : 0)  << 16;
    d |= uint32_t((N >> 3) & 0x3F) << 17;
    d |= uint32_t((M >> 4) & 0x1F) << 24;
    return d;
}

// ============== tcgen05 fence / commit ==============
__device__ __forceinline__ void tcgen05_fence_before_mma() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_mma() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_commit(Mbarrier& bar) {
    uint32_t addr = smem_to_uint(&bar.raw);
    asm volatile(
        "tcgen05.commit.cta_group" TCGEN05_CTA_GROUP ".mbarrier::arrive::one.b64 [%0];"
        :: "r"(addr));
}

// ============== one-time setup ==============
__device__ __forceinline__
void prefetch_tma_descriptors(const HcaParams& p) {
    if ((threadIdx.x & 31) == 0 && (threadIdx.x / 32) == 0) {
        prefetch_tma_descriptor(&p.tma_Q_sw128);
        prefetch_tma_descriptor(&p.tma_Q_sw64);
        prefetch_tma_descriptor(&p.tma_O);
        prefetch_tma_descriptor(&p.tma_Kc);
        prefetch_tma_descriptor(&p.tma_Kc_scales);
        prefetch_tma_descriptor(&p.tma_Kc_rope);
        prefetch_tma_descriptor(&p.tma_Kswa);
        prefetch_tma_descriptor(&p.tma_Kswa_scales);
        prefetch_tma_descriptor(&p.tma_Kswa_rope);
    }
}

// ============== KernelState (per-CTA derived info, not in HcaParams) ==============
struct KernelState {
    int batch_idx;
    int head_half_idx;       // 0 or 1 (H=128 → 2× B_H=64)
    int partition_idx;       // split-K index along compressed leg

    int compressed_start;    // K-token range owned by this CTA
    int compressed_end;
    int swa_start;
    int swa_end;

    float* partial_O;        // split-K partial output for this partition
    float* partial_lse;
};


__device__ __forceinline__
void query_tma(const HcaParams& p, const KernelState& ks, Smem& smem) {
    if (!elect_one_sync()) return;

    int q_tx = B_H * D_NOPE * sizeof(__nv_bfloat16)
             + B_H * D_ROPE * sizeof(__nv_bfloat16);
    mbar_expect(smem.bar_q_tma, q_tx);

    tma_load_3d(&p.tma_Q_sw128, ks.batch_idx, ks.head_half_idx, 0,
                smem.u.qo.q_sw128, smem.bar_q_tma);
    tma_load_3d(&p.tma_Q_sw64,  ks.batch_idx, ks.head_half_idx, 0,
                smem.u.qo.q_sw64,  smem.bar_q_tma);
}


__device__ __forceinline__
void init_state(KernelState& ks, const HcaParams& p) {
    ks.batch_idx        = blockIdx.y;
    ks.head_half_idx    = blockIdx.z;
    ks.partition_idx    = blockIdx.x;
    ks.compressed_start = 0;            // TODO: derive from host scheduler
    ks.compressed_end   = p.M_cur;
    ks.swa_start        = 0;
    ks.swa_end          = p.swa_len;
    ks.partial_O        = nullptr;
    ks.partial_lse      = nullptr;
}


// ============== tmem column layout ==============
struct tmem_cols {
    static constexpr int O      =   0;   // 256 cols  fp32  PV accumulator
    static constexpr int Q_NOPE = 256;   //  64 cols  fp8   QK A operand (nope)
    static constexpr int Q_ROPE = 320;   //  16 cols  bf16  QK A operand (rope)
    static constexpr int P      = 336;   //  32 cols  fp32  QK accumulator
    static constexpr int S      = 368;   //  16 cols  bf16  PV A operand (ts variant)
    // 384..512 = 128 cols spare
};


// ============== UTCCP: smem (Q) -> tmem ==============
// Uses tcgen05.cp.cta_group::1.<shape>; the CTA's elected thread issues all copies.
//
// REVIEW: the PTX shape `.128x128b` covers 128 lanes × 128 bits per copy. With M=64
// the canonical SM100 path is `.64x128b.warpx2::02_13` which requires the issuing
// pair of warps to multicast; we use `.128x128b` here under the assumption that the
// downstream mma is M=64-half of an M=128 footprint and that lanes 64..127 are
// consumed by the peer half (or mirrored). If you stick with single-CTA M=64, switch
// to `.64x128b.warpx2::02_13` and have warps 0/2 (or 1/3) co-issue.
//
// REVIEW: dtype. The tmem_cols comments label Q_NOPE as fp8 but q_sw128 is bf16. We
// preserve the smem layout and copy the bf16 bytes; whoever consumes Q_NOPE through
// kind::f8f6f4 mma must either reinterpret-cast or run a quant pass. If QK is meant
// to run as kind::f16, the fp8 label on Q_NOPE is wrong — relabel it as 256 cols bf16
// and update the K-block stride below from 16 to 32 cols.
__device__ __forceinline__
void utccp_q_to_tmem(const Smem& smem) {
    if (!elect_one_sync()) return;

    const uint32_t base = smem.tmem_start_addr;

    // Q nope: 64 rows × 512 bf16 in 128B-swizzle smem -> tmem cols [Q_NOPE, Q_NOPE+64).
    // Each .128x128b copy moves 128 lanes × 128b = 16B per lane = 8 bf16 per lane,
    // and lands in 4 tmem cols. 64 cols / 4 = 16 ops.
    {
        uint32_t s_addr = smem_to_uint(&smem.u.qo.q_sw128[0]);
        // 128B swizzle, K-major bf16: stride from 8 rows to next 8 rows = 8 * 512 * 2 = 8192 B.
        // Leading-dim offset is unused under swizzled layouts (encoded as 0).
        uint64_t desc = make_smem_desc(s_addr,
                                       /*leading_byte_off=*/ 0,
                                       /*stride_byte_off =*/ 8 * D_NOPE * sizeof(__nv_bfloat16),
                                       /*base_offset     =*/ 0,
                                       /*swizzle_mode    =*/ 2 /*128B*/);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t taddr  = tmem_addr(base, tmem_cols::Q_NOPE + i * 4);
            uint64_t d_step = desc + (uint64_t((i * 16 * D_NOPE * sizeof(__nv_bfloat16)) & 0x3FFFFu) >> 4);
            asm volatile(
                "tcgen05.cp.cta_group" TCGEN05_CTA_GROUP ".128x128b [%0], %1;"
                :: "r"(taddr), "l"(d_step));
        }
    }

    // Q rope: 64 rows × 64 bf16 in 64B-swizzle smem -> tmem cols [Q_ROPE, Q_ROPE+16).
    // .128x128b: 16 cols / 4 = 4 ops.
    {
        uint32_t s_addr = smem_to_uint(&smem.u.qo.q_sw64[0]);
        uint64_t desc = make_smem_desc(s_addr,
                                       /*leading_byte_off=*/ 0,
                                       /*stride_byte_off =*/ 8 * D_ROPE * sizeof(__nv_bfloat16),
                                       /*base_offset     =*/ 0,
                                       /*swizzle_mode    =*/ 4 /*64B*/);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t taddr  = tmem_addr(base, tmem_cols::Q_ROPE + i * 4);
            uint64_t d_step = desc + (uint64_t((i * 16 * D_ROPE * sizeof(__nv_bfloat16)) & 0x3FFFFu) >> 4);
            asm volatile(
                "tcgen05.cp.cta_group" TCGEN05_CTA_GROUP ".128x128b [%0], %1;"
                :: "r"(taddr), "l"(d_step));
        }
    }

    tcgen05_fence_after_mma();   // make tmem writes visible to subsequent mma issues
}


// ============== score mma: rope (bf16) ==============
// Q_rope(tmem) [64,64] · K_rope(smem)[64,64]^T -> P(tmem) [64,64]
// init=true zeroes P before this issue (rope mma fires first per tile).
__device__ __forceinline__
void tcgen05_mma_qk_rope(const Smem& smem, int buf, bool init = true) {
    const uint32_t base = smem.tmem_start_addr;
    const uint32_t d_addr = tmem_addr(base, tmem_cols::P);
    const uint32_t a_addr = tmem_addr(base, tmem_cols::Q_ROPE);

    // K rope smem: bf16 [TILE_KV, D_ROPE] = [64, 64], 64B swizzle, K-major.
    uint32_t b_smem = smem_to_uint(&smem.u.kv.rope[buf][0]);
    uint64_t b_desc = make_smem_desc(b_smem,
                                     /*leading_byte_off=*/ 0,
                                     /*stride_byte_off =*/ 8 * D_ROPE * sizeof(__nv_bfloat16),
                                     /*base_offset     =*/ 0,
                                     /*swizzle_mode    =*/ 4 /*64B*/);

    uint32_t idesc = make_idesc(/*M=*/64, /*N=*/64,
                                /*atype=BF16*/ 1, /*btype=BF16*/ 1,
                                /*transA=*/ false, /*transB=*/ false);

    int enable_input_d = init ? 0 : 1;   // when init=true, compute D = A*B (no add)

    tcgen05_fence_before_mma();
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "setp.ne.s32 P, %4, 0;\n"
        "tcgen05.mma.cta_group" TCGEN05_CTA_GROUP ".kind::f16 "
            "[%0], [%1], %2, %3, P;\n"
        "}\n"
        :: "r"(d_addr), "r"(a_addr), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
    );
}


// ============== score mma: nope (fp8 f8f6f4, ONE K=128 issue) ==============
// Q_nope(tmem) [64,128] · K_nope(smem)[64,128]^T -> P(tmem) [64,64]  (accumulate)
// k_block ∈ {0,1,2,3} covers full K=512 in 4× K=128 chunks.
//
// REVIEW: tmem A column stride per K-block is 16 here (matches the existing call
// site `Q_NOPE + k_block*16`). With kind::f8f6f4 K=128 fp8, A in tmem occupies
// K_bits/32 = 32 cols typically; the stride-16 layout assumes K=64 fp8 packed via
// `.pack::16b` semantics or a custom column packing. Verify against the layout
// table for M=64 non-WS once you settle on the Q quant path.
__device__ __forceinline__
void tcgen05_mma_qk_nope(const Smem& smem, int buf, int k_block) {
    const uint32_t base = smem.tmem_start_addr;
    const uint32_t d_addr = tmem_addr(base, tmem_cols::P);
    const uint32_t a_addr = tmem_addr(base, tmem_cols::Q_NOPE + k_block * 16);

    // K nope smem: fp8 [TILE_KV, D_NOPE], 128B swizzle, K-major.
    // Byte offset for this K-block of 128 fp8 (= 128 B per row).
    const int b_offset = k_block * 128;
    uint32_t b_smem = smem_to_uint(
        reinterpret_cast<const char*>(&smem.u.kv.latent[buf][0]) + b_offset);
    uint64_t b_desc = make_smem_desc(b_smem,
                                     /*leading_byte_off=*/ 0,
                                     /*stride_byte_off =*/ 8 * D_NOPE * sizeof(__nv_fp8_e4m3),
                                     /*base_offset     =*/ 0,
                                     /*swizzle_mode    =*/ 2 /*128B*/);

    // atype = btype = E4M3 (= 0)
    uint32_t idesc = make_idesc(/*M=*/64, /*N=*/64,
                                /*atype=E4M3*/ 0, /*btype=E4M3*/ 0,
                                /*transA=*/ false, /*transB=*/ false);

    // Always accumulate into P (rope already initialized it).
    tcgen05_fence_before_mma();
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "setp.ne.s32 P, %4, 0;\n"
        "tcgen05.mma.cta_group" TCGEN05_CTA_GROUP ".kind::f8f6f4 "
            "[%0], [%1], %2, %3, P;\n"
        "}\n"
        :: "r"(d_addr), "r"(a_addr), "l"(b_desc), "r"(idesc), "r"(/*enable_input_d=*/1)
    );
}


// ============== column-wise e8m0 scale fold on P ==============
// After each fp8 nope mma, fold the per-token e8m0 scale into the fp32 P
// accumulator (column-wise multiply). 4 scales per token live in
// smem.u.kv.scales[buf]; this applies the one for `k_block`.
//
// Shape: P is [M=64, N=TILE_KV=64] fp32 in tmem at col `tmem_cols::P` (32 cols).
// Each warp issues a tcgen05.ld.32x32b spanning 32 lanes × 32b per col. We use
// .x32 to read all 32 tmem cols of P in 32 b32 regs, multiply lane-broadcast
// e8m0 → fp32 scales, and store back.
//
// REVIEW: this version assumes the elected warp owns all 64 lanes of P in two
// .32x32b passes. For the correct collective layout you may need 4 warps
// cooperating across lane groups — verify when you fill in softmax.
__device__ __forceinline__
void apply_p_scale_fold(const Smem& smem, int buf, int k_block) {
    // Warp-collective reads/writes; require warp-uniform `taddr`.
    const uint32_t base = smem.tmem_start_addr;
    const uint32_t lane = threadIdx.x & 31;

    // Per-token e8m0 scale: scales[buf][token * SCALES_PER_TOKEN + k_block].
    // e8m0 → fp32: byte is the unsigned biased exponent; mantissa = 1.0.

    // Two .32x32b passes covering 64 lanes × 32 cols of P.
    // Pass 0: lanes [0,32),  Pass 1: lanes [32,64).
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        uint32_t taddr = tmem_addr(base, tmem_cols::P, /*lane=*/pass * 32);
        // Load 32 cols (.x32) of fp32 from P fragment.
        uint32_t r[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r[0]),  "=r"(r[1]),  "=r"(r[2]),  "=r"(r[3]),
              "=r"(r[4]),  "=r"(r[5]),  "=r"(r[6]),  "=r"(r[7]),
              "=r"(r[8]),  "=r"(r[9]),  "=r"(r[10]), "=r"(r[11]),
              "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
              "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
              "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
              "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
              "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
            : "r"(taddr)
        );

        // Each tcgen05.ld.32x32b lane holds 1 fp32 per col across 32 cols.
        // The N-column index for this lane is `pass*32 + (col_in_frag)` — but the
        // per-token scale we computed is for token = lane*2+j. Both indexings need
        // to agree; if the fp32 P fragment is laid out as col=token then `r[i]` for
        // this lane corresponds to a single token across i. In that case scale by
        // a per-i scalar, not per-j. The simpler safe version: rebroadcast scales
        // across registers using the e8m0 byte that matches the column.
        // REVIEW: reconfirm P fragment col-↔-token mapping for M=64 layout F.

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int token = pass * 32 + i;   // assumed col-i = token-(pass*32+i)
            __nv_fp8_e8m0 s8 = smem.u.kv.scales[buf][token * SCALES_PER_TOKEN + k_block];
            uint32_t bits = uint32_t(reinterpret_cast<const uint8_t&>(s8)) << 23;
            float s = __int_as_float(int(bits));
            float v = __int_as_float(int(r[i]));
            r[i] = uint32_t(__float_as_int(v * s));
        }

        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
            :: "r"(r[0]),  "r"(r[1]),  "r"(r[2]),  "r"(r[3]),
               "r"(r[4]),  "r"(r[5]),  "r"(r[6]),  "r"(r[7]),
               "r"(r[8]),  "r"(r[9]),  "r"(r[10]), "r"(r[11]),
               "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
               "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]),
               "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
               "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
               "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]),
               "r"(taddr)
        );
    }

    (void)lane;
}


// ============== value mma: ts variant ==============
// S(tmem) [64,64] · V(smem)[64,256] -> O(tmem) [64,256]  for one half of D_V.
// Caller invokes twice with v_smem_off ∈ {0, 256} (in elements) and o_col_off ∈ {0, 128}.
//
// REVIEW: V is fp8 in cache (smem.u.kv.latent); kind::f16 needs bf16 B. Either run
// kind::f8f6f4 here too (with S also fp8-quantized in tmem) or stage V to bf16 first.
__device__ __forceinline__
void tcgen05_mma_pv_ts(const Smem& smem, int buf,
                       int v_smem_off, int o_col_off,
                       bool init = false) {
    const uint32_t base = smem.tmem_start_addr;
    const uint32_t d_addr = tmem_addr(base, tmem_cols::O + o_col_off);
    const uint32_t a_addr = tmem_addr(base, tmem_cols::S);

    // V smem: same buffer as K nope; treat as [TILE_KV, D_V] starting at v_smem_off.
    // For PV with V used as KxN (K=TILE_KV, N=256), V is K-major in the same physical
    // layout as K, just consumed differently. The smem desc encodes V at the offset.
    uint32_t v_smem = smem_to_uint(
        reinterpret_cast<const char*>(&smem.u.kv.latent[buf][0])
        + v_smem_off * sizeof(__nv_fp8_e4m3));
    uint64_t b_desc = make_smem_desc(v_smem,
                                     /*leading_byte_off=*/ 0,
                                     /*stride_byte_off =*/ 8 * D_NOPE * sizeof(__nv_fp8_e4m3),
                                     /*base_offset     =*/ 0,
                                     /*swizzle_mode    =*/ 2 /*128B*/);

    uint32_t idesc = make_idesc(/*M=*/64, /*N=*/256,
                                /*atype=BF16*/ 1, /*btype=BF16*/ 1,
                                /*transA=*/ false, /*transB=*/ false);

    int enable_input_d = init ? 0 : 1;

    tcgen05_fence_before_mma();
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "setp.ne.s32 P, %4, 0;\n"
        "tcgen05.mma.cta_group" TCGEN05_CTA_GROUP ".kind::f16 "
            "[%0], [%1], %2, %3, P;\n"
        "}\n"
        :: "r"(d_addr), "r"(a_addr), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
    );
}


__device__ __forceinline__
void init_smem(Smem& smem) {
    if (threadIdx.x == 0) {
        mbar_init(smem.bar_q_tma,      1);
        mbar_init(smem.bar_q_utccp,    1);
        mbar_init(smem.bar_last_store, 128);

        #pragma unroll
        for (int i = 0; i < NUM_BUFS; ++i) {
            mbar_init(smem.bar_rope_ready[i], 1);
            mbar_init(smem.bar_raw_ready [i], 1);
            mbar_init(smem.bar_qk_done   [i], 1);
            mbar_init(smem.bar_so_ready  [i], 128);
            mbar_init(smem.bar_sv_done   [i], 1);
        }

        mbar_init(smem.bar_cprss_in,   1);
        mbar_init(smem.bar_cprss_done, 1);
        mbar_init(smem.bar_leg_merge,  128);
    }
    __syncthreads();
}

}
