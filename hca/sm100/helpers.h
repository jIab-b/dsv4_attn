#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cstdint>

#include "../params.h"

namespace dsv4::hca::sm100 {

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
constexpr int NUM_THREADS = 384;               // 2 warpgroups: softmax, mma+producers (no dequant — fp8 mma direct)

// ============== Mbarrier (storage only; logic in free functions) ==============
struct Mbarrier {
    uint64_t raw;
};

__device__ __forceinline__
void mbar_init(Mbarrier& bar, int count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.raw));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbar_arrive(Mbarrier& bar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.raw));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"(addr));
}

__device__ __forceinline__
void mbar_expect(Mbarrier& bar, int tx_bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.raw));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbar_wait(Mbarrier& bar, int phase) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.raw));
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
    extern __shared__ alignas(16) char __smem_storage[];
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
    if (!cute::elect_one_sync()) return;

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
// Issues a burst of SM100_UTCCP_128dp256bit_1cta copies that collectively
// move Q_nope (64 x 512 fp8 in q_sw128) -> tmem[Q_NOPE..Q_NOPE+64)
// and    Q_rope (64 x  64 bf16 in q_sw64) -> tmem[Q_ROPE..Q_ROPE+16)
__device__ __forceinline__
void utccp_q_to_tmem(const Smem& smem) {
    // TODO: 16 utccp ops over q_sw128 -> tmem cols Q_NOPE..Q_NOPE+64
    // TODO: 2  utccp ops over q_sw64  -> tmem cols Q_ROPE..Q_ROPE+16
}


// ============== score mma: rope (bf16 ts) ==============
// Q_rope(tmem) [64,64] · K_rope(smem)[64,64]^T -> P(tmem) [64,64]
// `init=true` zeroes P before this issue; the rope mma fires first per tile so init=true.
__device__ __forceinline__
void tcgen05_mma_qk_rope(const Smem& smem, int buf, bool init = true) {
    // TODO: tcgen05.mma.cta_group::1.kind::f16  M=64 N=64 K=64
    //         d  = tmem[P..P+32)
    //         a  = tmem[Q_ROPE..Q_ROPE+16)
    //         b  = smem desc for smem.u.kv.dequant[buf].rope (SW64)
    //         idesc encodes init flag
}


// ============== score mma: nope (fp8 ts, ONE K=128 issue) ==============
// Q_nope(tmem) [64,128] · K_nope(smem)[64,128]^T -> P(tmem) [64,64]  (accum)
// Caller iterates k_block ∈ {0,1,2,3} to cover full K=512.
__device__ __forceinline__
void tcgen05_mma_qk_nope(const Smem& smem, int buf, int k_block) {
    // TODO: tcgen05.mma.cta_group::1.kind::f8f6f4  M=64 N=64 K=128
    //   d  = tmem[P..P+32)
    //   a  = tmem[Q_NOPE + k_block*16 .. +16)
    //   b  = smem desc for smem.u.kv.raw_latent[buf] at byte offset k_block*128
    //   idesc with init=false (rope mma initialized P)
}


// ============== column-wise e8m0 scale fold on P ==============
// After each fp8 nope mma, fold the per-token e8m0 scale into the fp32 P
// accumulator (column-wise multiply). 4 scales per token live in
// smem.u.kv.scales[buf]; this applies the one for `k_block`.
__device__ __forceinline__
void apply_p_scale_fold(const Smem& smem, int buf, int k_block) {
    // TODO: tcgen05.ld P fragments,
    //       multiply column t of P by cvt_e8m0_to_bf16(smem.scales[buf][t][k_block]),
    //       tcgen05.st back to P.
}


// ============== value mma: ts variant ==============
// S(tmem) [64,64] · V(smem)[64,256]      -> O(tmem) [64,256]   for one half of D_V
// Caller invokes twice with v_smem_off ∈ {0, 256} and o_col_off ∈ {0, 128}.
__device__ __forceinline__
void tcgen05_mma_pv_ts(const Smem& smem, int buf,
                       int v_smem_off, int o_col_off,
                       bool init = false) {
    // TODO: tcgen05.mma.cta_group::1.kind::f16  M=64 N=256 K=64
    //         d  = tmem[O + o_col_off .. + 128)
    //         a  = tmem[S..S+16)
    //         b  = smem desc for smem.u.kv.dequant[buf].latent at byte offset v_smem_off*sizeof(bf16)
    //         idesc encodes init flag (true only on the very first PV of the kernel)
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
