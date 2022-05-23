#!/usr/bin/env python3
import os

import torch  # pylint: disable=unused-import
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ["-std=c++14"]

CC = os.getenv("CC", None)
nvcc_args = [
    "-ccbin=" + CC,
    "-gencode",
    "arch=compute_50,code=sm_50",
    "-gencode",
    "arch=compute_52,code=sm_52",
    "-gencode",
    "arch=compute_60,code=sm_60",
    "-gencode",
    "arch=compute_61,code=sm_61",
    "-gencode",
    "arch=compute_70,code=sm_70",
    "-gencode",
    "arch=compute_70,code=compute_70",
]

setup(
    name="channelnorm_cuda",
    ext_modules=[
        CUDAExtension(
            "channelnorm_cuda",
            ["channelnorm_cuda.cc", "channelnorm_kernel.cu"],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ],
    cmdclass={
        #'build_ext': BuildExtension
        "build_ext": BuildExtension.with_options(use_ninja=False)
    },
)
