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
constexpr int NUM_THREADS = 384;               // 3 warpgroups: softmax, mma+producers, dequant

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
            // dequantized (bf16) tiles fed to mma
            struct {
                __nv_bfloat16 latent[TILE_KV * D_NOPE];           // 64 KB  fed K-side and V-side
                __nv_bfloat16 rope  [TILE_KV * D_ROPE];           //  8 KB
            } dequant[NUM_BUFS];                                  // 144 KB total

            // raw fp8 staging from TMA, before dequant WG converts -> dequant.latent
            __nv_fp8_e4m3  raw_latent[NUM_BUFS][TILE_KV * D_NOPE];     // 64 KB
            __nv_fp8_e8m0  scales    [NUM_BUFS][TILE_KV * SCALES_PER_TOKEN]; // 0.5 KB
        } kv;
    } u;

    __nv_bfloat16 s_buf      [B_H * TILE_KV];                     // 8 KB   S/P scratch
    float         p_exchange [4][16 * TILE_KV / 4];               // 4 KB   warp coord
    float         rowwise_max[128];                               // 0.5 KB
    uint32_t      tmem_start_addr;

    // mbarriers (inline)
    Mbarrier bar_q_tma;
    Mbarrier bar_q_utccp;
    Mbarrier bar_last_store;

    Mbarrier bar_latent_ready[NUM_BUFS];   // dequantized latent ready for mma
    Mbarrier bar_rope_ready  [NUM_BUFS];
    Mbarrier bar_raw_ready   [NUM_BUFS];   // raw fp8 landed (producer → dequant WG)
    Mbarrier bar_raw_free    [NUM_BUFS];   // raw fp8 buf reusable (dequant WG → producer)

    Mbarrier bar_qk_done [NUM_BUFS];
    Mbarrier bar_so_ready[NUM_BUFS];
    Mbarrier bar_sv_done [NUM_BUFS];

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


__device__ __forceinline__
void init_smem(Smem& smem) {
    if (threadIdx.x == 0) {
        mbar_init(smem.bar_q_tma,      1);
        mbar_init(smem.bar_q_utccp,    1);
        mbar_init(smem.bar_last_store, 128);

        #pragma unroll
        for (int i = 0; i < NUM_BUFS; ++i) {
            mbar_init(smem.bar_latent_ready[i], 1);
            mbar_init(smem.bar_rope_ready  [i], 1);
            mbar_init(smem.bar_raw_ready   [i], 1);
            mbar_init(smem.bar_raw_free    [i], 128);
            mbar_init(smem.bar_qk_done     [i], 1);
            mbar_init(smem.bar_so_ready    [i], 128);
            mbar_init(smem.bar_sv_done     [i], 1);
        }

        mbar_init(smem.bar_cprss_in,   1);
        mbar_init(smem.bar_cprss_done, 1);
        mbar_init(smem.bar_leg_merge,  128);
    }
    __syncthreads();
}

}
