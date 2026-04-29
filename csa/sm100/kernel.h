#pragma once

#include "../params.h"
#include <cuda_runtime.h>

namespace dsv4::csa::sm100 {

// SM100 (Blackwell) launcher. Defined in kernel.cuh, instantiated per
// M in instantiations/m{M}.cu. Runtime → constexpr dispatch on `p.m`
// lives in csa/interface.h.
template<int M>
void launch_csa_compress_reduce(const CsaCompressReduceParams& p, cudaStream_t stream);

extern template void launch_csa_compress_reduce<4>(const CsaCompressReduceParams&, cudaStream_t);

}  // namespace dsv4::csa::sm100
