# fabq_rc_cuda

Native-quantized inference CUDA extension for **FABQ-RC** (Fisher-Adaptive
Binary Quantization with Residual Codebooks).

The forward pass operates directly on the compressed FABQ-RC format
(int4 channels + bit-packed binary channels + k-means codebook indices).
**The FP16 weight matrix is never materialized** at inference time.

## What's in here

- `src/fabq_rc_format.h` - the on-disk / in-memory format spec
- `src/fabq_rc_gemm.cu` - the CUDA kernels (int4-only, binary-only, mixed)
- `src/fabq_rc_quant.cpp` - CPU-side quantization + file I/O
- `src/bindings.cpp` - pybind11 glue
- `quantized_linear.py` - the `QuantizedLinear` nn.Module
- `model.py` - `quantize_model_in_place` for swapping layers
- `io.py` - thin Python wrappers around the C++ file I/O
- `kmeans.py` - shared k-means codebook builder
- `tests/test_kernel.py` - numerical correctness vs PyTorch reference

## Build

```bash
# From this directory:
pip install -e .

# Or in-place build (no install):
python setup.py build_ext --inplace
```

Build requirements:
- PyTorch >= 2.0 with CUDA support
- CUDA toolkit (nvcc) >= 11.8
- C++17 compiler
- pybind11 >= 2.10
- NVIDIA GPU with compute capability >= 7.0 (Volta or newer). The v2
  W4A16 tensor-core path (fp16x fp16 WMMA m16n16k16) needs SM 7.0+; the
  scalar v2 paths work on SM 7.5+. Anything from V100 up is supported.
  Note: this is W4A16, not native int4 TC - the activations stay fp16
  because int4xint4 (W4A4) explodes PPL at FABQ-RC's 1.21 bpw weight quant.

## Quick test

```bash
cd tests
python test_kernel.py
```

The CUDA tests are skipped if no GPU is available. The I/O round-trip tests
work on any platform.

## Why a custom kernel

Most quantization repos (AWQ, GPTQ, BitsAndBytes) "dequantize" the weights
to FP16 just-in-time before the matmul, which means peak memory during
inference is the FP16 weight size (~25 GB for 12B), not the compressed
size (~1.2 GB for FABQ-RC).

This extension reads the compressed buffers directly. Peak inference memory
is ~1.2 GB for the weights + activations. The trade-off is v1 is scalar
(no tensor cores), so it's slower than cuBLAS at dense matmul. v2 will
add `mma.sync` for the int4 submatrix.

## Format

Each layer's compressed representation is:
- `int4_channels: int64[n_int4]` - which output channels are int4
- `int4_weights: int8[n_int4, in_features]` - the int4 values (stored as int8)
- `int4_scales: fp16[n_int4]` - per-row scale
- `binary_channels: int64[n_binary]` - which output channels are binary
- `binary_bits: uint8` - packed, 1 bit per (channel, input)
- `binary_scales: fp16[n_binary, n_blocks]` - per-block scale
- `codebook_idx: uint8[n_binary, n_blocks]` - index into the shared codebook
- `codebook: fp16[n_clusters, max_blocksize]` - shared across all layers

See `src/fabq_rc_format.h` for the on-disk binary layout.

## License

Apache 2.0
