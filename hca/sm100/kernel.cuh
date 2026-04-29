#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../params.h"

namespace dsv4::hca::sm100 {

template<int M_PRIME, int THREADS>
__global__ void hca_compress_kernel(HcaCompressReduceParams p);

}
