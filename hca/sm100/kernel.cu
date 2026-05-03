#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "kernel.h"
#include "helpers.h"

namespace dsv4::hca::sm100 {

///*****************************************************************************
///*** kernel component includes
///*****************************************************************************

#include "tmem_wg.cu"
#include "tma_load.cu"
#include "mma.cu"

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
    Smem& smem = shared_state();

    prefetch_tma_descriptors(p);
    init_smem(smem);

    KernelState ks;
    init_state(ks, p);

    if (ks.wg == 0) query_load(p, ks, smem);
    __syncthreads();

    if (ks.wg == 0) {
        softmax_warpgroup(p, ks, smem);
    } else if (ks.wg == 1) {
        switch (ks.warp_idx) {
            case 4: mma_warp             (p, ks, smem); break;
            case 5: nope_prod_warp       (p, ks, smem); break;
            case 6: rope_prod_warp       (p, ks, smem); break;
            case 7: compress_branch_warp (p, ks, smem); break;
        }
    }
}

///*****************************************************************************
///*** end global kernel
///*****************************************************************************

}
