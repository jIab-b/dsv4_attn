#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace dsv4::hca::sm100 {

struct HcaCombineParams {
    int B;
    int H;
    int D;                  // must be 512
    int num_splits;

    const float* __restrict__ partial_O;      // [split, B, H, D]
    const float* __restrict__ partial_lse;    // [split, B, H], log2
    __nv_bfloat16* __restrict__ O;            // [B, H, D]
    float* __restrict__ lse;                  // [B, H], natural log, optional
    const float* __restrict__ attn_sink;      // [H], optional, natural-log units

    int64_t stride_partial_O_split;
    int64_t stride_partial_O_b;
    int64_t stride_partial_O_h;
    int64_t stride_partial_lse_split;
    int64_t stride_partial_lse_b;
    int64_t stride_O_b;
    int64_t stride_O_h;
    int64_t stride_lse_b;
};

void launch_hca_combine(const HcaCombineParams& p, cudaStream_t stream);

}  // namespace dsv4::hca::sm100

