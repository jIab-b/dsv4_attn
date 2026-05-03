#pragma once

#include "../params.h"
#include <cuda_runtime.h>

namespace dsv4::hca::sm100 {

void launch_hca_decode(const HcaParams& p);

}
