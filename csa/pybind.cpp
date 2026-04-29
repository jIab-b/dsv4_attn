#include <pybind11/pybind11.h>

#include "interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DSV4 CSA compressor — Pass-2 reduce";
    m.def("csa_compress_reduce_fwd",
          &dsv4::csa::csa_compress_reduce_fwd,
          "CSA compressor reduce (Eqs 11-12 of DSV4 §2.3.1)",
          pybind11::arg("C_a"),
          pybind11::arg("C_b"),
          pybind11::arg("Z_a"),
          pybind11::arg("Z_b"),
          pybind11::arg("B_a"),
          pybind11::arg("B_b"),
          pybind11::arg("m"));
}
