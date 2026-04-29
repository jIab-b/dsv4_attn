// CSA Pass-2 reduce kernel — m=4 instantiation (Blackwell / SM100).
// V4-Flash and V4-Pro both fix m=4 (paper §4.2.1).
// c_out is dynamic (handled inside the kernel via tid stride),
// so this single instantiation covers c_out ∈ {512, 128, 640}.

#include "../kernel.cuh"

namespace dsv4::csa::sm100 {

template void launch_csa_compress_reduce<4>(const CsaCompressReduceParams&, cudaStream_t);

}  // namespace dsv4::csa::sm100
