"""Setup script for the fabq_rc_cuda extension.

Builds the C++/CUDA extension in-place. After `pip install -e .` (or
`python setup.py build_ext --inplace`), the module is importable as
`fabq_rc_cuda` from anywhere.

Build requirements:
- PyTorch >= 2.0 with CUDA support
- CUDA toolkit (nvcc) >= 11.8
- C++17 compiler
- pybind11 >= 2.10

Compute capability:
- WMMA m16n16k16 tensor-core path requires SM 7.0+ (Volta).
- The scalar v2 paths require SM 7.5+ (Turing).
- Default arch list covers Volta / Turing / Ampere / Ada / Hopper / Blackwell.
- Override with the TORCH_CUDA_ARCH_LIST env var for a specific GPU.

Kernels are not A100-specific - they target any NVIDIA GPU with compute
capability >= 7.0. A100/H100/B200 benefit from the tensor-core path;
V100/T4/RTX 30xx fall through to the vectorized scalar path automatically.
"""

import os
from setuptools import setup, Extension

# Detect CUDA_HOME
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
if not os.path.exists(CUDA_HOME):
    for candidate in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    ]:
        if os.path.exists(candidate):
            CUDA_HOME = candidate
            break

this_dir = os.path.dirname(os.path.abspath(__file__))

# Default arch list. Covers Volta (7.0) -> Blackwell (10.0). The v2 tensor
# core path uses WMMA m16n16k16 which works on all of these. If you only
# care about one arch (faster build), set TORCH_CUDA_ARCH_LIST externally.
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = (
        "7.0;7.5;8.0;8.6;8.9;9.0;10.0"
    )

ext = Extension(
    name="fabq_rc_cuda._C",
    sources=[
        os.path.join(this_dir, "src", "bindings.cpp"),
        os.path.join(this_dir, "src", "fabq_rc_quant.cpp"),
        os.path.join(this_dir, "src", "fabq_rc_gemm.cu"),       # v1: reference
        os.path.join(this_dir, "src", "fabq_rc_gemm_v2.cu"),    # v2: production
    ],
    include_dirs=[os.path.join(this_dir, "src")],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-Xcompiler", "/O2",
        ],
    },
    define_macros=[("WITH_CUDA", "1")],
    libraries=["c10", "torch", "torch_cpu", "torch_cuda"],
)

setup(
    name="fabq_rc_cuda",
    version="0.2.0",
    description="FABQ-RC native-quantized inference (CUDA extension)",
    ext_modules=[ext],
    cmdclass={},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
