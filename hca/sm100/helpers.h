#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cstdint>
#include <type_traits>

#include "../params.h"

namespace dsv4::hca::sm100 {

///*****************************************************************************
///*** data / structures
///*****************************************************************************

constexpr int B_H         = 64;
constexpr int TILE_KV     = 128;
constexpr int NUM_BUFS    = 2;
constexpr int D_NOPE      = 448;               // = HEAD_DIM(512) - D_ROPE(64), per vLLM
constexpr int D_ROPE      = 64;
constexpr int D_V         = 512;               // value-MMA M-axis cover; V is K-nope (448) zero-padded to 512
constexpr int D_Q         = D_NOPE + D_ROPE;   // 512
constexpr int M_PRIME     = 128;
constexpr int W_SWA       = 128;
constexpr int QUANT_TILE  = 64;                // e8m0 group: 1 scale / 64 channels
constexpr int SCALES_PER_TOKEN = D_NOPE / QUANT_TILE;  // 7 (cache pads to 8 on disk)
constexpr int NUM_THREADS = 256;               // 2 warpgroups: softmax, mma+producers
constexpr int MMA_MXF8_K  = 32;
constexpr int MMA_BF16_K  = 16;                // kind::f16 cta_group::2 dense K
constexpr int SCORE_M     = TILE_KV;           // KV tokens
constexpr int SCORE_N     = B_H;               // heads
constexpr int VALUE_M     = 256;               // value dims per value-MMA chunk (V padded to 512)
constexpr int VALUE_N     = B_H;               // heads
constexpr int SCORE_K_BLOCKS = D_NOPE / MMA_MXF8_K;            // 14
constexpr int ROPE_K_BLOCKS  = D_ROPE / MMA_BF16_K;            //  4
constexpr int VALUE_DIM_BLOCKS = D_V / VALUE_M;                // 2
constexpr int VALUE_TOKEN_BLOCKS = TILE_KV / MMA_MXF8_K;       // 4

// tcgen05 cta_group used kernel-wide (must be the same for every tcgen05 op)
#define TCGEN05_CTA_GROUP "::2"

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
            __nv_fp8_e4m3 latent[NUM_BUFS][TILE_KV * D_NOPE];               // 128 KB fp8 K/V staging
            __nv_fp8_e8m0 scales[NUM_BUFS][TILE_KV * SCALES_PER_TOKEN];     //   1 KB e8m0 per-64 scales
            __nv_bfloat16 rope  [NUM_BUFS][TILE_KV * D_ROPE];               //  32 KB bf16 rope
            __nv_fp8_e4m3 softmax[NUM_BUFS][B_H * TILE_KV];                 //  16 KB fp8 S^T for value MMA
        } kv;
    } u;

    float         p_exchange [4][16 * TILE_KV / 4];   // 8 KB  warp coord
    float         partial_m  [4][32];                  // 0.5 KB  local warp max partials
    float         partial_l  [4][32];                  // 0.5 KB  local warp sum partials
    float         peer_partial_m[4][32];               // 0.5 KB  peer CTA warp max partials
    float         peer_partial_l[4][32];               // 0.5 KB  peer CTA warp sum partials
    float         rowwise_max[128];                   // 0.5 KB
    float         rolling_m[B_H];                     // committed softmax max per head row
    float         rolling_l[B_H];                     // committed softmax sum per head row
    float         curr_m[B_H];                        // current-tile softmax max per head row
    float         curr_l[B_H];                        // current-tile softmax sum per head row
    uint8_t       value_mma_scales[NUM_BUFS][2][B_H]; // e8m0 per (buf, warp, head) for fp8 P (value MMA B)
    uint32_t      tmem_start_addr;

    // mbarriers (inline)
    uint64_t mbar_q_tma;
    uint64_t mbar_q_utccp;
    uint64_t mbar_last_store;

    uint64_t mbar_rope_ready[NUM_BUFS];
    uint64_t mbar_raw_ready [NUM_BUFS];   // fp8 latent + scales landed
    uint64_t mbar_qk_done   [NUM_BUFS];
    uint64_t mbar_p_consumed[NUM_BUFS];   // softmax wg done reading P TMEM
    uint64_t mbar_so_ready  [NUM_BUFS];
    uint64_t mbar_sv_done   [NUM_BUFS];   // also serves as "latent buf free" for nope_prod

    // chunked epilogue handshakes (see epi_wg.cuh): one pair per value-dim chunk
    uint64_t mbar_so_ready_chunk[VALUE_DIM_BLOCKS][NUM_BUFS];
    uint64_t mbar_sv_done_chunk [VALUE_DIM_BLOCKS][NUM_BUFS];
    uint64_t mbar_alpha_ready   [NUM_BUFS];

    // softmax warpgroup row-half exchange
    uint64_t mbar_softmax;
    uint64_t mbar_peer_partials;
};

struct KernelState {
    int batch_idx;
    int head_half_idx;       // 0 or 1 (H=128 -> 2 x B_H=64)
    int partition_idx;       // split-K index along compressed leg

    int wg;                  // = threadIdx.x / 128  (0..1)
    int warp_idx;            // = threadIdx.x / 32   (global, 0..7)
    int lane;                // = threadIdx.x & 31
    int cta_rank_in_pair;    // = %cluster_ctarank & 1 for tcgen05 cta_group::2

    int compressed_start;    // K-token range owned by this CTA
    int compressed_end;
    int swa_start;
    int swa_end;

    float* partial_O;        // split-K partial output for this partition
    float* partial_lse;

    // partial_O write-column per (chunk, half) — set by host/init layer
    int    partial_O_out_col[VALUE_DIM_BLOCKS][2];
};

struct tmem_cols {
    static constexpr int O              =   0;   // 2 x 64-col fp32 value accum chunks
    static constexpr int P              = 128;   // 64-col fp32 score tile: [128 kv, 64 head]
    static constexpr int SCORE_SCALE_A  = 192;   // e8m0 K scales in TMEM sub-columns
    static constexpr int SCORE_SCALE_B  = 208;   // e8m0 Q scales in TMEM sub-columns
    static constexpr int VALUE_SCALE_A  = 224;   // e8m0 V scales; see value-scale caveat
    static constexpr int VALUE_SCALE_B  = 240;   // e8m0 S scales
};

///*****************************************************************************
///*** end data / structures
///*****************************************************************************

///*****************************************************************************
///*** mbar helpers
///*****************************************************************************

__device__ __forceinline__
uint32_t smem_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void mbar_init(uint64_t& mbar, int count) {
    uint32_t addr = smem_to_uint(&mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbar_arrive(uint64_t& mbar) {
    uint32_t addr = smem_to_uint(&mbar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"(addr));
}

__device__ __forceinline__
void mbar_expect(uint64_t& mbar, int tx_bytes) {
    uint32_t addr = smem_to_uint(&mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes));
}

// 500 ms budget at ~2 GHz boost on B200; trap if we wait longer.
#ifndef HCA_MBAR_TIMEOUT_CYCLES
#define HCA_MBAR_TIMEOUT_CYCLES 1000000000ULL
#endif

__device__ __forceinline__
void mbar_wait(uint64_t& mbar, int phase) {
    uint32_t addr = smem_to_uint(&mbar);
    int done = 0;
    uint64_t start = clock64();
    while (!done) {
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
            "selp.b32 %0, 1, 0, P;\n"
            "}\n"
            : "=r"(done) : "r"(addr), "r"(phase)
        );
        if (!done && (clock64() - start) > HCA_MBAR_TIMEOUT_CYCLES) {
            __trap();
        }
    }
}

__device__ __forceinline__
uint64_t mbar_arrive_state(uint64_t& mbar) {
    uint32_t addr = smem_to_uint(&mbar);
    uint64_t state;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
                 : "=l"(state) : "r"(addr) : "memory");
    return state;
}

__device__ __forceinline__
void mbar_wait_state(uint64_t& mbar, uint64_t state) {
    uint32_t addr = smem_to_uint(&mbar);
    int done = 0;
    uint64_t start = clock64();
    while (!done) {
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "mbarrier.try_wait.acquire.cluster.shared::cta.b64 P, [%1], %2;\n"
            "selp.b32 %0, 1, 0, P;\n"
            "}\n"
            : "=r"(done) : "r"(addr), "l"(state) : "memory"
        );
        if (!done && (clock64() - start) > HCA_MBAR_TIMEOUT_CYCLES) {
            __trap();
        }
    }
}

__device__ __forceinline__
void sync_softmax_wg(Smem& smem) {
    uint64_t state = mbar_arrive_state(smem.mbar_softmax);
    mbar_wait_state(smem.mbar_softmax, state);
}

static constexpr uint64_t PEER_ADDR_MASK = 1ull << 24;

template <typename T>
__device__ __forceinline__
T* peer_ptr(T* p) {
    return reinterpret_cast<T*>(reinterpret_cast<uint64_t>(p) ^ PEER_ADDR_MASK);
}

__device__ __forceinline__
void dsmem_expect_tx(uint64_t* peer_mbar, int tx_bytes) {
    uint64_t addr = reinterpret_cast<uint64_t>(peer_mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cluster.b64 _, [%0], %1;"
        :: "l"(addr), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__
void dsmem_copy_async(
    void* dst_peer_smem,
    const void* src_smem,
    int bytes,
    uint64_t* peer_mbar
) {
    uint32_t dst  = smem_to_uint(dst_peer_smem);
    uint32_t src  = smem_to_uint(src_smem);
    uint32_t mbar = smem_to_uint(peer_mbar);

    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];"
        :: "r"(dst), "r"(src), "r"(bytes), "r"(mbar) : "memory");
}

__device__ __forceinline__
void dsmem_wait(uint64_t& mbar) {
    uint64_t state = mbar_arrive_state(mbar);
    mbar_wait_state(mbar, state);
}

///*****************************************************************************
///*** end mbar helpers
///*****************************************************************************

///*****************************************************************************
///*** other tcgen / ptx helpers
///*****************************************************************************

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

__device__ __forceinline__ Smem& init_smem() {
    extern __shared__ char __smem_storage[];
    Smem& smem = *reinterpret_cast<Smem*>(__smem_storage);
    if (threadIdx.x == 0) {
        mbar_init(smem.mbar_q_tma,      1);
        mbar_init(smem.mbar_q_utccp,    1);
        mbar_init(smem.mbar_last_store, 128);
        #pragma unroll
        for (int i = 0; i < NUM_BUFS; ++i) {
            mbar_init(smem.mbar_rope_ready[i], 1);
            mbar_init(smem.mbar_raw_ready [i], 1);
            mbar_init(smem.mbar_qk_done   [i], 1);
            mbar_init(smem.mbar_p_consumed[i], 128);
            mbar_init(smem.mbar_so_ready  [i], 128);
            mbar_init(smem.mbar_sv_done   [i], 1);
            mbar_init(smem.mbar_alpha_ready[i], 1);
            #pragma unroll
            for (int c = 0; c < VALUE_DIM_BLOCKS; ++c) {
                mbar_init(smem.mbar_so_ready_chunk[c][i], 128);
                mbar_init(smem.mbar_sv_done_chunk [c][i], 1);
            }
        }
        mbar_init(smem.mbar_softmax,       128);
        mbar_init(smem.mbar_peer_partials, 129);
    }
    __syncthreads();
    return smem;
}

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
// `mbar_expect` on `mbar` with the correct tx-byte count before issuing this.
template<typename T>
__device__ __forceinline__
void tma_load_3d(const CUtensorMap* desc,
                 int crd0, int crd1, int crd2,
                 T* dst_smem, uint64_t& mbar) {
    uint32_t s_addr = smem_to_uint(dst_smem);
    uint32_t b_addr = smem_to_uint(&mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(s_addr), "l"(desc), "r"(b_addr),
          "r"(crd0), "r"(crd1), "r"(crd2)
        : "memory");
}

// 32-bit TMEM address: [31:16] lane index, [15:0] column index.
// `base` is the per-CTA tmem allocation base (col=0, lane=0).
__device__ __forceinline__
uint32_t tmem_addr(uint32_t base, int col, int lane = 0) {
    return base + (uint32_t(lane) << 16) + uint32_t(col);
}

// 256 columns covers O(128) + P(64) + 4×scale(16) = 256.
constexpr int TMEM_NUM_COLS = 512;

__device__ __forceinline__
void tcgen05_alloc(uint32_t* smem_dst, int n_cols) {
    uint32_t addr = smem_to_uint(smem_dst);
    asm volatile(
        "tcgen05.alloc.cta_group" TCGEN05_CTA_GROUP ".sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(addr), "n"(TMEM_NUM_COLS) : "memory");
    (void)n_cols;
}

__device__ __forceinline__
void tcgen05_relinquish_alloc_permit() {
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group" TCGEN05_CTA_GROUP ".sync.aligned;"
        ::: "memory");
}

__device__ __forceinline__
void tcgen05_dealloc(uint32_t base, int n_cols) {
    asm volatile(
        "tcgen05.dealloc.cta_group" TCGEN05_CTA_GROUP ".sync.aligned.b32 %0, %1;"
        :: "r"(base), "n"(TMEM_NUM_COLS) : "memory");
    (void)n_cols;
}

__device__ __forceinline__ void tcgen05_fence_before_mma() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_mma() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit(uint64_t& mbar) {
    uint32_t addr = smem_to_uint(&mbar);
    asm volatile(
        "tcgen05.commit.cta_group" TCGEN05_CTA_GROUP ".mbarrier::arrive::one.b64 [%0];"
        :: "r"(addr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__
void tcgen05_ld_32x32b_x32(uint32_t taddr, float (&out)[32]) {
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
    tcgen05_wait_ld();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        out[i] = __uint_as_float(r[i]);
    }
}

__device__ __forceinline__
void tcgen05_st_32x32b_x32(uint32_t taddr, const float (&in)[32]) {
    uint32_t r[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) r[i] = __float_as_uint(in[i]);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
        " %17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
        :: "r"(taddr),
           "r"(r[0]),  "r"(r[1]),  "r"(r[2]),  "r"(r[3]),
           "r"(r[4]),  "r"(r[5]),  "r"(r[6]),  "r"(r[7]),
           "r"(r[8]),  "r"(r[9]),  "r"(r[10]), "r"(r[11]),
           "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
           "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]),
           "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
           "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
           "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
        : "memory"
    );
    tcgen05_wait_st();
}

///*****************************************************************************
///*** end other tcgen / ptx helpers
///*****************************************************************************

///*****************************************************************************
///*** quant / cvt helpers
///*****************************************************************************

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, s));
    return v;
}

// fp32 -> ue8m0 byte, round-positive (smallest power of 2 >= x).
__device__ __forceinline__ uint8_t f32_to_e8m0_rp(float x) {
    uint16_t pair;
    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %1;" : "=h"(pair) : "f"(x));
    return (uint8_t)(pair & 0xff);
}

// ue8m0 byte -> fp32 (= 2^(byte - 127)).
__device__ __forceinline__ float e8m0_to_f32(uint8_t b) {
    return __uint_as_float((uint32_t)b << 23);
}

// 4x fp32 -> 4x fp8 e4m3, RTNE saturating, packed into one u32.
__device__ __forceinline__ uint32_t f32x4_to_e4m3x4(const float (&v)[4]) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(v[0]), "f"(v[1]));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(v[2]), "f"(v[3]));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

///*****************************************************************************
///*** end quant / cvt helpers
///*****************************************************************************

///*****************************************************************************
///*** descriptor helpers
///*****************************************************************************

// SMEM descriptor (PTX 9.7.16.4.1, Table 40). 14-bit fields hold (x>>4) of
// 16-byte-aligned addresses/offsets. swizzle: 0=none, 2=128B, 4=64B, 6=32B.
__device__ __forceinline__
uint64_t make_smem_desc(
    const void* smem_ptr,
    uint32_t leading_byte_offset,
    uint32_t stride_byte_offset,
    uint32_t swizzle
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t d = 0;
    d |=  uint64_t((addr                  >> 4) & 0x3FFFu);
    d |= (uint64_t((leading_byte_offset   >> 4) & 0x3FFFu)) << 16;
    d |= (uint64_t((stride_byte_offset    >> 4) & 0x3FFFu)) << 32;
    d |=  uint64_t(1) << 46;                          // fixed 0b001
    d |= (uint64_t(swizzle) & 0x7u) << 61;
    return d;
}

__device__ __forceinline__
uint64_t make_kmajor_smem_desc(const void* ptr, uint32_t row_stride_elems) {
    return make_smem_desc(ptr, /*leading=*/0,
                          /*stride=*/8u * row_stride_elems,
                          /*swizzle 32B=*/6u);
}

// 64-byte swizzle variant — matches the TMA descriptors used for Q-rope and
// K-rope tiles (CU_TENSOR_MAP_SWIZZLE_64B).
__device__ __forceinline__
uint64_t make_kmajor_smem_desc_sw64(const void* ptr, uint32_t row_stride_elems) {
    return make_smem_desc(ptr, /*leading=*/0,
                          /*stride=*/8u * row_stride_elems,
                          /*swizzle 64B=*/4u);
}

// Instruction descriptor for kind::mxf8f6f4 (PTX 9.7.16.4.2, Table 43).
// atype/btype = e4m3 (=0), scale type = ue8m0 (=1), dense, no transpose,
// SFA_ID = SFB_ID = 0 (caller iterates K-blocks via separate idesc updates).
template<int M, int N>
__device__ __forceinline__
uint32_t make_mxf8_idesc() {
    static_assert(M == 128 || M == 256, "mxf8f6f4 2SM requires M in {128,256}");
    static_assert(N % 8 == 0 && N <= 256, "N must be multiple of 8, <=256");
    constexpr uint32_t n_field = (N >> 3) & 0x3F;     // bits 17-22
    constexpr uint32_t m_field = (M >> 7) & 0x3;      // bits 27-28
    return (n_field << 17) | (1u << 23) | (m_field << 27);
}

// Instruction descriptor for kind::f16 with bf16 A/B and fp32 D (PTX 9.7.16.4.2,
// Table 42). Dense, no negate. TransA / TransB flip the descriptor's K-major
// vs MN-major selector (bits 15/16) — leave both false when the contiguous dim
// of the smem layout is the MMA's K dim.
template<int M, int N, bool TransA = false, bool TransB = false>
__device__ __forceinline__
uint32_t make_bf16_idesc() {
    static_assert(M == 128 || M == 256, "kind::f16 cta_group::2 requires M in {128,256}");
    static_assert(N >= 16 && N <= 256 && (N % 16) == 0,
                  "kind::f16 cta_group::2 requires N a multiple of 16 in [16, 256]");
    constexpr uint32_t dtype   = 1u;                  // F32 (bits  4- 5)
    constexpr uint32_t atype   = 1u;                  // BF16 (bits 7- 9)
    constexpr uint32_t btype   = 1u;                  // BF16 (bits 10-12)
    constexpr uint32_t trans_a = TransA ? 1u : 0u;    // bit 15
    constexpr uint32_t trans_b = TransB ? 1u : 0u;    // bit 16
    constexpr uint32_t n_field = (N >> 3) & 0x3Fu;    // bits 17-22
    constexpr uint32_t m_field = (M >> 4) & 0x1Fu;    // bits 24-28
    return (dtype   <<  4)
         | (atype   <<  7)
         | (btype   << 10)
         | (trans_a << 15)
         | (trans_b << 16)
         | (n_field << 17)
         | (m_field << 24);
}

// Score: A=K[token, dim], B=Q^T[dim, head], D=P[token, head].
__device__ __forceinline__
uint64_t make_score_a_sdesc(const Smem& smem, int buf, int k_block) {
    const auto* ptr = &smem.u.kv.latent[buf][k_block * MMA_MXF8_K];
    return make_kmajor_smem_desc(ptr, D_NOPE);
}

__device__ __forceinline__
uint64_t make_score_b_sdesc(const Smem& smem, int k_block) {
    const auto* q_fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(&smem.u.qo.o_buf[0]);
    return make_kmajor_smem_desc(
        &q_fp8[k_block * MMA_MXF8_K], D_NOPE);
}

__device__ __forceinline__
uint32_t make_score_idesc() { return make_mxf8_idesc<SCORE_M, SCORE_N>(); }

// Score-rope: A=K_rope[token, rope], B=Q_rope^T[rope, head], D=P[token, head].
// Both operands are bf16 in smem with the rope axis (= K) contiguous, so the
// descriptors are K-major. The rope tiles are loaded with SWIZZLE_64B, so the
// sdesc must request 64B swizzling too.
__device__ __forceinline__
uint64_t make_score_rope_a_sdesc(const Smem& smem, int buf, int k_block) {
    const auto* ptr = &smem.u.kv.rope[buf][k_block * MMA_BF16_K];
    return make_kmajor_smem_desc_sw64(ptr, D_ROPE);
}

__device__ __forceinline__
uint64_t make_score_rope_b_sdesc(const Smem& smem, int k_block) {
    const auto* ptr = &smem.u.qo.q_sw64[k_block * MMA_BF16_K];
    return make_kmajor_smem_desc_sw64(ptr, D_ROPE);
}

__device__ __forceinline__
uint32_t make_score_rope_idesc() { return make_bf16_idesc<SCORE_M, SCORE_N>(); }

// Value: A=V^T[value_dim, token], B=S[token, head], D=O^T[value_dim, head].
// `latent` must be staged in V^T order before this descriptor is correct.
__device__ __forceinline__
uint64_t make_value_a_sdesc(const Smem& smem, int buf, int dim_block, int token_block) {
    const int value_dim = dim_block * VALUE_M;
    const int token     = token_block * MMA_MXF8_K;
    const auto* ptr = &smem.u.kv.latent[buf][value_dim * TILE_KV + token];
    return make_kmajor_smem_desc(ptr, TILE_KV);
}

__device__ __forceinline__
uint64_t make_value_b_sdesc(const Smem& smem, int buf, int token_block) {
    const int token = token_block * MMA_MXF8_K;
    const auto* ptr = &smem.u.kv.softmax[buf][token];
    return make_kmajor_smem_desc(ptr, TILE_KV);
}

__device__ __forceinline__
uint32_t make_value_idesc() { return make_mxf8_idesc<VALUE_M, VALUE_N>(); }

__device__ __forceinline__
uint32_t score_scale_a_tmem(const Smem& smem) {
    return tmem_addr(smem.tmem_start_addr, tmem_cols::SCORE_SCALE_A);
}

__device__ __forceinline__
uint32_t score_scale_b_tmem(const Smem& smem) {
    return tmem_addr(smem.tmem_start_addr, tmem_cols::SCORE_SCALE_B);
}

__device__ __forceinline__
uint32_t value_scale_a_tmem(const Smem& smem) {
    return tmem_addr(smem.tmem_start_addr, tmem_cols::VALUE_SCALE_A);
}

__device__ __forceinline__
uint32_t value_scale_b_tmem(const Smem& smem) {
    return tmem_addr(smem.tmem_start_addr, tmem_cols::VALUE_SCALE_B);
}

///*****************************************************************************
///*** end descriptor helpers
///*****************************************************************************

///*****************************************************************************
///*** mma helpers
///*****************************************************************************

// Inline tcgen05.mma SS form for kind::mxf8f6f4 with block_scale scale_vec::1X
// (PTX 9.7.16.10.9.1). Both A and B come from SMEM via 64-bit descriptors;
// scales come from TMEM at scale_{a,b}_tmem.
template<int M, int N>
__device__ __forceinline__
void tcgen05_mxf8_2sm_ss(
    uint32_t d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t scale_a_tmem,
    uint32_t scale_b_tmem,
    bool accumulate
) {
    const uint32_t idesc = make_mxf8_idesc<M, N>();
    const uint32_t enable_in = accumulate ? 1u : 0u;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.u32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n"
        "}\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"(scale_a_tmem), "r"(scale_b_tmem), "r"(enable_in)
        : "memory");
}

// A thin 2SM mxf8 score issue. The caller must have staged K/Q scale factors
// into SCORE_SCALE_A/B before issuing this K=32 block.
__device__ __forceinline__
void score_mma(const Smem& smem, int buf, int k_block, bool accumulate) {
    const uint32_t d_tmem = tmem_addr(smem.tmem_start_addr, tmem_cols::P);
    const uint64_t a_desc = make_score_a_sdesc(smem, buf, k_block);
    const uint64_t b_desc = make_score_b_sdesc(smem, k_block);

    tcgen05_fence_before_mma();
    tcgen05_mxf8_2sm_ss<SCORE_M, SCORE_N>(
        d_tmem, a_desc, b_desc,
        score_scale_a_tmem(smem), score_scale_b_tmem(smem),
        accumulate);
}

// Inline tcgen05.mma SS form for kind::f16 with bf16 A/B and fp32 D
// (PTX 9.7.16.10.9.1). disable_output_lane is forced to all-zero (zero in
// register `z`, repeated eight times for cta_group::2); the M extent in the
// idesc dictates which lanes are actually written.
template<int M, int N>
__device__ __forceinline__
void tcgen05_bf16_2sm_ss(
    uint32_t d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    bool accumulate
) {
    const uint32_t idesc = make_bf16_idesc<M, N>();
    const uint32_t enable_in = accumulate ? 1u : 0u;
    asm volatile(
        "{\n"
        ".reg .b32 z;\n"
        "mov.b32 z, 0;\n"
        ".reg .pred p;\n"
        "setp.ne.u32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 "
        "[%0], %1, %2, %3, {z, z, z, z, z, z, z, z}, p;\n"
        "}\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_in)
        : "memory");
}

// Rope contribution to the score tile. Accumulates into the same P TMEM region
// as score_mma; the caller must have issued the nope k-blocks first (with
// accumulate=false on kb=0) so this MMA folds rope on top with accumulate=true.
__device__ __forceinline__
void score_mma_rope(const Smem& smem, int buf, int k_block, bool accumulate) {
    const uint32_t d_tmem = tmem_addr(smem.tmem_start_addr, tmem_cols::P);
    const uint64_t a_desc = make_score_rope_a_sdesc(smem, buf, k_block);
    const uint64_t b_desc = make_score_rope_b_sdesc(smem, k_block);

    tcgen05_fence_before_mma();
    tcgen05_bf16_2sm_ss<SCORE_M, SCORE_N>(
        d_tmem, a_desc, b_desc, accumulate);
}

// Value uses the same SS form. This assumes V is transposed/staged as
// [value_dim, token] and softmax is staged as S^T [head, token].
__device__ __forceinline__
void value_mma(const Smem& smem, int buf, int dim_block, int token_block, bool accumulate) {
    const uint32_t d_tmem = tmem_addr(
        smem.tmem_start_addr, tmem_cols::O + dim_block * VALUE_N);
    const uint64_t a_desc = make_value_a_sdesc(smem, buf, dim_block, token_block);
    const uint64_t b_desc = make_value_b_sdesc(smem, buf, token_block);

    tcgen05_fence_before_mma();
    tcgen05_mxf8_2sm_ss<VALUE_M, VALUE_N>(
        d_tmem, a_desc, b_desc,
        value_scale_a_tmem(smem), value_scale_b_tmem(smem),
        accumulate);
}

///*****************************************************************************
///*** end mma helpers
///*****************************************************************************

///*****************************************************************************
///*** init helpers
///*****************************************************************************

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

__device__ __forceinline__
int get_cta_rank_in_pair() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return int(rank & 1u);
}

__device__ __forceinline__
void init_state(KernelState& ks, const HcaParams& p, Smem& smem) {
    ks.batch_idx        = blockIdx.y;
    ks.head_half_idx    = blockIdx.z;
    ks.partition_idx    = blockIdx.x;
    ks.wg               = threadIdx.x / 128;
    ks.warp_idx         = threadIdx.x / 32;
    ks.lane             = threadIdx.x & 31;
    ks.cta_rank_in_pair = get_cta_rank_in_pair();
    ks.compressed_start = 0;            // TODO: derive from host scheduler
    ks.compressed_end   = p.M_cur;
    ks.swa_start        = 0;
    ks.swa_end          = p.swa_len;
    ks.partial_O        = p.partial_O == nullptr ? nullptr :
        p.partial_O
        + int64_t(ks.partition_idx) * p.stride_partial_O_split
        + int64_t(ks.batch_idx)     * p.stride_partial_O_b
        + int64_t(ks.head_half_idx) * B_H * p.stride_partial_O_h;
    ks.partial_lse      = p.partial_lse == nullptr ? nullptr :
        p.partial_lse
        + int64_t(ks.partition_idx) * p.stride_partial_lse_split
        + int64_t(ks.batch_idx)     * p.stride_partial_lse_b
        + int64_t(ks.head_half_idx) * B_H;
    if (ks.warp_idx == 0) {
        tcgen05_alloc(&smem.tmem_start_addr, TMEM_NUM_COLS);
        tcgen05_relinquish_alloc_permit();
    }
}


///*****************************************************************************
///*** end init helpers
///*****************************************************************************

}
