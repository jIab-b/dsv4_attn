#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

#include "kernel.h"
#include "helpers.h"
#include "combine.h"

namespace dsv4::hca::sm100 {

///*****************************************************************************
///*** kernel component includes
///*****************************************************************************

#include "tmem_wg.cuh"
#include "tma_load.cuh"
#include "mma.cuh"

///*****************************************************************************
///*** end kernel component includes
///*****************************************************************************


///*****************************************************************************
///*** compressor / legacy stubs
///*****************************************************************************

__device__ __inline__ void compress_branch_warp(const HcaParams&, const KernelState&, Smem&) {}

template<int CPRSS_NUM>
__device__ __inline__ void hca_compress(
    const HcaParams& p,
    Smem& smem,
    int b,
    int channel_s
) {
    const __nv_bfloat16* base_C    = p.C    + b * p.stride_C_b + channel_s;
    const __nv_bfloat16* base_Z    = p.Z    + b * p.stride_C_b + channel_s;
    const __nv_bfloat16* base_bias = p.bias                    + channel_s;
          __nv_fp8_e4m3* base_out  = p.Kc   + b * p.stride_Kc_b
                                            + p.M_cur * p.stride_Kc_n
                                            + channel_s;
          __nv_fp8_e8m0* base_scales = p.Kc_scales + b * p.stride_Kc_scales_b
                                                  + p.M_cur * p.stride_Kc_scales_n
                                                  + channel_s / QUANT_TILE;

    constexpr int NUM_SCALES = (CPRSS_NUM + QUANT_TILE - 1) / QUANT_TILE;
    const __nv_fp8_e8m0 one_scale(1.0f);
    for (int s = threadIdx.x; s < NUM_SCALES; s += blockDim.x) {
        base_scales[s] = one_scale;
    }

    for (int c = threadIdx.x; c < CPRSS_NUM; c += blockDim.x) {
        float m   = -INFINITY;
        float l   = 0.f;
        float acc = 0.f;

        for (int i = 0; i < M_PRIME; ++i) {
            const int row = i * p.stride_C_n;
            float zi = (float)ldg_bf16(base_Z    + row + c)
                     + (float)ldg_bf16(base_bias + row + c);
            float ci = (float)ldg_bf16(base_C    + row + c);

            float m_new = fmaxf(m, zi);
            float alpha = __expf(m  - m_new);
            float beta  = __expf(zi - m_new);
            l   = l   * alpha + beta;
            acc = acc * alpha + beta * ci;
            m   = m_new;
        }

        base_out[c] = __nv_fp8_e4m3(acc / l);
    }
}

///*****************************************************************************
///*** end compressor / legacy stubs
///*****************************************************************************


///*****************************************************************************
///*** global kernel
///*****************************************************************************

__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
hca_decode_kernel(__grid_constant__ const HcaParams p) {

    
    prefetch_tma_descriptors(p);
    Smem smem; init_smem(&smem);

    KernelState ks;
    init_state(ks, p);

    __syncthreads();

    if (ks.wg == 0) query_load(p, ks, smem);
    __syncthreads();

    if (ks.wg == 0) {
        softmax_warpgroup(p, ks, smem);
    } else if (ks.wg == 1) {
        epi_wg(p, ks, smem);
    } else if (ks.wg == 2) {
        switch (ks.warp_idx) {
            case 4: mma_warp             (p, ks, smem); break;
            case 5: nope_prod_warp       (p, ks, smem); break;
            case 6: rope_prod_warp       (p, ks, smem); break;
            case 7: compress_branch_warp (p, ks, smem); break;
        }
    }

    __syncthreads();
    if (ks.warp_idx == 0 && ks.lane == 0) {
        tcgen05_dealloc(smem.tmem_start_addr, TMEM_NUM_COLS);
    }
}

///*****************************************************************************
///*** end global kernel
///*****************************************************************************


///*****************************************************************************
///*** host launcher
///*****************************************************************************

namespace {

inline CUtensorMapDataType cu_map_bf16() { return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; }
inline CUtensorMapDataType cu_map_u8()   { return CU_TENSOR_MAP_DATA_TYPE_UINT8; }

// Encode a 3D TMA descriptor.
//
// Convention used by tma_load_3d below: crd0 = batch, crd1 = head/token,
// crd2 = inner dim. The driver's globalDim[0] is the innermost-stride axis,
// so we feed it innermost-first and order the crds accordingly.
inline CUresult encode_3d(
    CUtensorMap& desc,
    void* gmem,
    CUtensorMapDataType dtype,
    uint64_t inner_dim,         // axis 0 (innermost element dim)
    uint64_t mid_dim,           // axis 1
    uint64_t outer_dim,         // axis 2
    uint64_t stride_axis1_b,    // bytes between consecutive slices along axis 1
    uint64_t stride_axis2_b,    // bytes between consecutive slices along axis 2
    uint32_t box_inner,
    uint32_t box_mid,
    uint32_t box_outer,
    CUtensorMapSwizzle swizzle
) {
    uint64_t sizes[3]   = {inner_dim, mid_dim, outer_dim};
    uint64_t strides[2] = {stride_axis1_b, stride_axis2_b};
    uint32_t box[3]     = {box_inner, box_mid, box_outer};
    uint32_t elem_str[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        &desc, dtype, 3, gmem, sizes, strides, box, elem_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

constexpr uint64_t bf16_b = sizeof(__nv_bfloat16);

}  // namespace

void launch_hca_decode(const HcaParams& p_in) {
    HcaParams p = p_in;

    const int B    = p.B;
    const int H    = 128;
    const int M_max = std::max(p.M_cur, 1);
    const int W    = std::max(p.swa_len, 1);

    // Q [B, H, D_Q]   bf16; box loads B_H heads × D_NOPE/D_ROPE.
    encode_3d(p.tma_Q_sw128, p.Q, cu_map_bf16(),
              D_NOPE, H, B,
              D_Q * bf16_b, int64_t(H) * D_Q * bf16_b,
              D_NOPE, B_H, 1,
              CU_TENSOR_MAP_SWIZZLE_128B);
    encode_3d(p.tma_Q_sw64, reinterpret_cast<__nv_bfloat16*>(p.Q) + D_NOPE,
              cu_map_bf16(),
              D_ROPE, H, B,
              D_Q * bf16_b, int64_t(H) * D_Q * bf16_b,
              D_ROPE, B_H, 1,
              CU_TENSOR_MAP_SWIZZLE_64B);

    // O [B, H, D_V]   bf16
    encode_3d(p.tma_O, p.O, cu_map_bf16(),
              D_V, H, B,
              D_V * bf16_b, int64_t(H) * D_V * bf16_b,
              D_V, B_H, 1,
              CU_TENSOR_MAP_SWIZZLE_128B);

    // K compressed [B, M_max, D_NOPE]  fp8 (uint8 carrier) + scales [B, M_max, SCALES_PER_TOKEN] u8
    // + rope [B, M_max, D_ROPE] bf16
    encode_3d(p.tma_Kc, p.Kc, cu_map_u8(),
              D_NOPE, M_max, B,
              D_NOPE, int64_t(M_max) * D_NOPE,
              D_NOPE, TILE_KV, 1,
              CU_TENSOR_MAP_SWIZZLE_NONE);
    encode_3d(p.tma_Kc_scales, p.Kc_scales, cu_map_u8(),
              SCALES_PER_TOKEN, M_max, B,
              SCALES_PER_TOKEN, int64_t(M_max) * SCALES_PER_TOKEN,
              SCALES_PER_TOKEN, TILE_KV, 1,
              CU_TENSOR_MAP_SWIZZLE_NONE);
    encode_3d(p.tma_Kc_rope, p.Kc_rope, cu_map_bf16(),
              D_ROPE, M_max, B,
              D_ROPE * bf16_b, int64_t(M_max) * D_ROPE * bf16_b,
              D_ROPE, TILE_KV, 1,
              CU_TENSOR_MAP_SWIZZLE_64B);

    encode_3d(p.tma_Kswa, p.Kswa, cu_map_u8(),
              D_NOPE, W, B,
              D_NOPE, int64_t(W) * D_NOPE,
              D_NOPE, TILE_KV, 1,
              CU_TENSOR_MAP_SWIZZLE_NONE);
    encode_3d(p.tma_Kswa_scales, p.Kswa_scales, cu_map_u8(),
              SCALES_PER_TOKEN, W, B,
              SCALES_PER_TOKEN, int64_t(W) * SCALES_PER_TOKEN,
              SCALES_PER_TOKEN, TILE_KV, 1,
              CU_TENSOR_MAP_SWIZZLE_NONE);
    encode_3d(p.tma_Kswa_rope, p.Kswa_rope, cu_map_bf16(),
              D_ROPE, W, B,
              D_ROPE * bf16_b, int64_t(W) * D_ROPE * bf16_b,
              D_ROPE, TILE_KV, 1,
              CU_TENSOR_MAP_SWIZZLE_64B);

    const size_t smem_bytes = sizeof(Smem);
    cudaFuncSetAttribute(
        hca_decode_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes));

    dim3 grid(/*splits*/ 1, /*B*/ B, /*head_halves*/ 2);
    dim3 block(NUM_THREADS, 1, 1);

    // 2-CTA cluster for tcgen05.cta_group::2. Pair along z (the head-halves
    // axis) so the cluster dim divides the grid (z=2 → 2 CTAs per cluster).
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim          = grid;
    cfg.blockDim         = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream           = 0;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 2};
    cfg.attrs    = attrs;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, hca_decode_kernel, p);
}

///*****************************************************************************
///*** end host launcher
///*****************************************************************************

}

///*****************************************************************************
///*** C ABI entry point — called via ctypes from python
///*****************************************************************************
extern "C" int hca_decode_fwd(
    const void* Q,                      // bf16 [B, n_h, c+n_rope], c = NoPE dim
    const void* Kc, const void* Kc_s, const void* Kc_rope,
    const void* Kswa, const void* Kswa_s, const void* Kswa_rope,
    const void* sink_logits,            // float [n_h] or null
    void* O,                            // bf16 [B, n_h, c+n_rope]
    void* partial_O, void* partial_lse, // float [1, B, n_h, c+n_rope], [1, B, n_h]
    float sm_scale,
    int B, int n_h, int c, int n_rope,
    int M_cur, int swa_len,
    void* stream                        // cudaStream_t or null
) {
    using namespace dsv4::hca;
    using namespace dsv4::hca::sm100;

    HcaParams pp{};
    pp.B = B;
    pp.M_cur = M_cur;
    pp.swa_len = swa_len;

    pp.Q  = (__nv_bfloat16*)Q;
    pp.O  = (__nv_bfloat16*)O;
    pp.Kc        = (__nv_fp8_e4m3*)Kc;
    pp.Kc_scales = (__nv_fp8_e8m0*)Kc_s;
    pp.Kc_rope   = (__nv_bfloat16*)Kc_rope;
    pp.Kswa        = (__nv_fp8_e4m3*)Kswa;
    pp.Kswa_scales = (__nv_fp8_e8m0*)Kswa_s;
    pp.Kswa_rope   = (__nv_bfloat16*)Kswa_rope;
    pp.partial_O   = (float*)partial_O;
    pp.partial_lse = (float*)partial_lse;
    pp.attn_sink   = (float*)sink_logits;

    const int d_q = c + n_rope;
    const int scale_blocks = c / 64;
    pp.stride_Q_b = (int64_t)n_h * d_q;
    pp.stride_O_b = (int64_t)n_h * d_q;
    pp.stride_Kc_b        = (int64_t)M_cur   * c;
    pp.stride_Kc_n        = c;
    pp.stride_Kc_scales_b = (int64_t)M_cur   * scale_blocks;
    pp.stride_Kc_scales_n = scale_blocks;
    pp.stride_Kc_rope_b   = (int64_t)M_cur   * n_rope;
    pp.stride_Kc_rope_n   = n_rope;
    pp.stride_Kswa_b        = (int64_t)swa_len * c;
    pp.stride_Kswa_n        = c;
    pp.stride_Kswa_scales_b = (int64_t)swa_len * scale_blocks;
    pp.stride_Kswa_scales_n = scale_blocks;
    pp.stride_Kswa_rope_b   = (int64_t)swa_len * n_rope;
    pp.stride_Kswa_rope_n   = n_rope;
    pp.stride_partial_O_split = (int64_t)B * n_h * d_q;
    pp.stride_partial_O_b     = (int64_t)n_h * d_q;
    pp.stride_partial_O_h     = d_q;
    pp.stride_partial_lse_split = (int64_t)B * n_h;
    pp.stride_partial_lse_b     = n_h;

    pp.sm_scale_log2 = sm_scale * 1.4426950408889634f;
    pp.rope_theta_compressed = 10000.0f;
    pp.rope_theta_swa        = 10000.0f;

    sm100::launch_hca_decode(pp);

    sm100::HcaCombineParams cp{};
    cp.B = B; cp.H = n_h; cp.D = d_q; cp.num_splits = 1;
    cp.partial_O   = (float*)partial_O;
    cp.partial_lse = (float*)partial_lse;
    cp.O           = (__nv_bfloat16*)O;
    cp.lse         = nullptr;
    cp.attn_sink   = (float*)sink_logits;
    cp.stride_partial_O_split   = pp.stride_partial_O_split;
    cp.stride_partial_O_b       = pp.stride_partial_O_b;
    cp.stride_partial_O_h       = pp.stride_partial_O_h;
    cp.stride_partial_lse_split = pp.stride_partial_lse_split;
    cp.stride_partial_lse_b     = pp.stride_partial_lse_b;
    cp.stride_O_b = (int64_t)n_h * d_q;
    cp.stride_O_h = d_q;
    cp.stride_lse_b = n_h;

    sm100::launch_hca_combine(cp, (cudaStream_t)stream);

    cudaError_t err = cudaGetLastError();
    return (int)err;
}
