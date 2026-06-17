"""Setup script for the fabq_rc_cuda extension.

Builds the C++/CUDA extension in-place. After `pip install -e .` (or
`python setup.py build_ext --inplace`), the module is importable as
`fabq_rc_cuda` from anywhere.

Build requirements:
- PyTorch >= 2.0 with CUDA support
- CUDA toolkit (nvcc) >= 11.8
- C++17 compiler
- pybind11 >= 2.10

Tested with CUDA 12.x on A100 80GB.
"""

from setuptools import setup, Extension
import os

# Detect CUDA_HOME
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
if not os.path.exists(CUDA_HOME):
    # Try common Windows locations
    for candidate in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    ]:
        if os.path.exists(candidate):
            CUDA_HOME = candidate
            break

this_dir = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    name="fabq_rc_cuda._C",
    sources=[
        os.path.join(this_dir, "src", "bindings.cpp"),
        os.path.join(this_dir, "src", "fabq_rc_quant.cpp"),
        os.path.join(this_dir, "src", "fabq_rc_gemm.cu"),
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
    version="0.1.0",
    description="FABQ-RC native-quantized inference (CUDA extension)",
    ext_modules=[ext],
    cmdclass={},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
