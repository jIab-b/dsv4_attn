#pragma once

#include "params.h"
#include <cuda_runtime.h>

namespace dsv4::hca {

template<int M_PRIME>
void launch_hca_compress_reduce(const HcaCompressReduceParams& p, cudaStream_t stream);

extern template void launch_hca_compress_reduce<128>(const HcaCompressReduceParams&, cudaStream_t);

}  
