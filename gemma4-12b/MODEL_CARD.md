---
license: apache-2.0
base_model: google/gemma-4-12B-it
tags:
  - quantization
  - 1-bit
  - fabq-rc
  - fisher-adaptive
  - gemma
  - native-quantized-inference
  - cuda-kernel
library_name: transformers
pipeline_tag: causal-lm
---

# Gemma 4 12B-it — FABQ-RC (Streaming + Native-Quantized Inference)

## What this is

A **native-quantized** version of Gemma 4 12B-it. The forward pass reads
the compressed FABQ-RC representation directly — no FP16 weight
materialization. Peak inference VRAM is **~2.2 GB** (vs 24 GB for BF16).

The quantization is built on the FABQ-RC method (Fisher-Adaptive Binary
Quantization with Residual Codebooks) — adaptive per-layer blocksize,
Fisher-weighted channel importance, mixed int4/binary precision, and a
shared k-means residual codebook for non-linear correction.

## Two variants in this folder

| Notebook | What it does | Peak VRAM | Speed |
|----------|-------------|-----------|-------|
| `FABQ-RC-Gemma4-12B.ipynb` | Text quantization, standard FP16 reconstruction | ~25 GB | cuBLAS speed |
| `streaming/FABQ-RC-Gemma4-12B-Streaming.ipynb` | Streaming + native-quantized inference | **~2.2 GB** | 5-10x slower (v1 scalar) |

The streaming variant requires a working CUDA toolchain to build the
`fabq_rc_cuda` extension. The standard variant only needs PyTorch.

## Quantization details

| Property | Value |
|----------|-------|
| **Base Model** | `google/gemma-4-12B-it` |
| **Bits per parameter** | ~1.21 bpw (theoretical) |
| **Quantization Method** | FABQ-RC |
| **Calibration Dataset** | C4 (2048 samples, seq_len=512) |
| **Precision Allocation** | Top 5% channels → int4, 95% → binary |
| **Blocksize Selection** | Adaptive per-layer {64, 128, 256, 512} |
| **Tied Embeddings** | Skipped (kept in BF16) |
| **Multimodal Encoders** | Skipped (text-only for v1) |

## Compressed size

| Component | Size |
|-----------|------|
| Quantized body (48 layers, FABQ-RC) | ~130 MB |
| Shared codebook | 256 KB |
| Tied embedding (BF16) | ~2 GB |
| **Total** | **~2.2 GB** |
| Compression ratio vs BF16 | **~11x** |

## The CUDA extension

`streaming/fabq_rc_cuda/` is a custom C++/CUDA extension that:

- Reads the FABQ-RC buffers directly during `forward()`
- Has three kernels: int4-only, binary-only, and mixed (the general case)
- Falls back to a PyTorch reference (which materializes FP16) when the
  extension isn't built
- Tested for numerical correctness against the PyTorch reference (see
  `streaming/fabq_rc_cuda/tests/test_kernel.py`)

v1 is scalar (no tensor cores). v2 will add `mma.sync` for the int4
submatrix. v1 is correct, not fast.

## Files in this folder

```
gemma4-12b/
├── FABQ-RC-Gemma4-12B.ipynb          # text-only quantization
├── build_notebook.py
├── README.md
├── MODEL_CARD.md                     # this file
├── FABQ_RC_GGUF_SPEC.md
└── streaming/
    ├── fabq_rc_cuda/                  # the CUDA extension
    ├── build_bucket.py                # one-time bucket build
    ├── build_streaming_notebook.py
    ├── FABQ-RC-Gemma4-12B-Streaming.ipynb
    └── STREAMING.md
```

## How to use the streaming variant

```bash
# 1. Build the bucket (one-time, requires A100 80GB)
cd streaming
python build_bucket.py --source google/gemma-4-12B-it --push

# 2. Run the streaming notebook
jupyter notebook FABQ-RC-Gemma4-12B-Streaming.ipynb
```

The bucket `toxzak/gemma-4-12B-it-fabq-rc-bucket` contains:
- The BF16 source shards (for the tied embedding)
- FABQ-RC stats (per-layer blocksize + int4/binary allocation)
- The shared k-means codebook
- Pre-quantized layer shards (one per decoder layer)

## Limitations

- **v1 is scalar, not tensor-core-accelerated.** Expect 5-10x slower
  than BF16 inference. v2 will close this gap.
- **Text-only for v1.** The vision and audio encoders are not quantized.
  If you need multimodal, the standard `FABQ-RC-Gemma4-12B.ipynb` notebook
  quantizes the model end-to-end (still uses the FP16 reconstruction
  path, but is multimodal-capable).
- **Tied embeddings stay in BF16** (~2 GB). Quantizing them is possible
  but out of scope for v1; see `../docs/specs/FABQ_RC_GGUF_SPEC.md` for the design.

## Citation

```bibtex
@misc{fabqrc2026,
    author = {Zach Maronek},
    title = {FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks},
    year = {2026},
    url = {https://github.com/toxzak/fabq-rc}
}
```

## Contact

- GitHub Issues: https://github.com/toxzak/fabq-rc/issues
- HuggingFace: https://huggingface.co/toxzak

---

*FABQ-RC by Zach Maronek · June 2026*
