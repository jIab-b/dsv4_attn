#include "../kernel.cuh"
#include "../kernel_splitk.cuh"

namespace dsv4::hca::sm100 {

template __global__ void hca_compress_kernel<128, 256>(HcaCompressReduceParams);
template __global__ void hca_compress_splitk_kernel<128, 256, 8>(HcaCompressReduceParams);

}
