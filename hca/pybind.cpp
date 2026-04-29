#include <pybind11/pybind11.h>

#include "interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DSV4 HCA compressor — Pass-2 reduce";
    m.def("hca_compress_reduce_fwd",
          &dsv4::hca::hca_compress_reduce_fwd,
          "HCA compressor reduce (Eqs 22-23 of DSV4 §2.3.2)",
          pybind11::arg("C"),
          pybind11::arg("Z"),
          pybind11::arg("bias"),
          pybind11::arg("m_prime"));
}
