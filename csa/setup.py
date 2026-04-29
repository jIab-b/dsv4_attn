"""Standalone build for the CSA compressor extension.

Independent of FlashMLA — this dir compiles by itself. To build:

    cd csa && pip install -e .

Or for an in-place build that test_csa.py picks up:

    cd csa && python setup.py build_ext --inplace
"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent

setup(
    name="dsv4_csa",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="dsv4_csa.cuda",
            sources=[
                str(HERE / "pybind.cpp"),
                str(HERE / "sm100" / "instantiations" / "m4.cu"),
            ],
            include_dirs=[str(HERE)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    "--use_fast_math",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    # Blackwell only.
                    "-gencode", "arch=compute_100f,code=sm_100f",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["dsv4_csa"],
    package_dir={"dsv4_csa": "."},
)
