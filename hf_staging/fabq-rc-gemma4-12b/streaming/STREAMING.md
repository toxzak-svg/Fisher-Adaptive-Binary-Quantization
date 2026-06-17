# FABQ-RC Streaming + Native-Quantized Inference — Design Doc

This document explains the architecture of the streaming variant of FABQ-RC
for Gemma 4 12B-it. Read this if you want to understand *why* the code is
structured the way it is, or if you're planning to extend it.

## The hard constraint: no FP16 weight materialization

The standard pattern in quantization repos (BitsAndBytes, AWQ, GPTQ) is:

1. Store weights in compressed form on disk
2. At inference, "dequantize" them to FP16 just-in-time
3. Use cuBLAS for the matmul
4. The peak VRAM during inference is the FP16 weight size

This works but defeats the point of quantization for memory-constrained
deployment. A 12B BF16 model is ~24 GB; a 12B FABQ-RC model is ~1.2 GB
in compressed form. The user wanted the inference-time memory to be the
compressed size, not the FP16 size.

That requires a custom kernel that reads the compressed buffers directly.

## The kernel design

`fabq_rc_cuda/src/fabq_rc_gemm.cu` has three kernels:

### 1. `fabq_rc_gemm_int4_kernel` — 100% int4 path

Reads `int4_weights: int8[n_int4, in_features]` and `int4_scales: fp16[n_int4]`.
Each block handles one (output_channel, batch_token) pair. Threads
cooperate over the input dimension, then warp-reduce to a single output.

This is the easy case — it can be replaced with `cuBLAS GEMM` after an
in-place dequantization if performance matters, but in v1 we just do scalar
multiplies.

### 2. `fabq_rc_gemm_binary_kernel` — 100% binary path

Reads the bit-packed `binary_bits` array, the per-block `binary_scales`,
the `codebook_idx` per block, and the shared `codebook` (fp16).

For each block:
- Unpack bits → ±1
- Multiply by per-block scale
- Add the codebook correction vector
- Multiply by activation, accumulate

This is the hard case. The bit unpacking is the bottleneck for memory
bandwidth; the codebook lookup is mostly L1-cache-friendly.

### 3. `fabq_rc_gemm_mixed_kernel` — general case

One block per output channel, dispatches int4 vs binary based on
`row_to_int4[o] >= 0` vs `row_to_binary[o] >= 0`. Used for every layer
in a FABQ-RC-quantized model (since each layer has ~5% int4 and 95%
binary).

## Memory layout (in-memory, not on-disk)

A `QuantizedLinear` module has these buffers, **and nothing else**. There
is no `self.weight` tensor:

```
self.int4_weights:   int8   [n_int4, in_features]
self.int4_scales:    fp16   [n_int4]
self.binary_bits:    uint8  [ceil(n_binary * in_features / 8)]   # packed
self.binary_scales:  fp16   [n_binary, n_blocks]
self.codebook_idx:   uint8  [n_binary, n_blocks]
self.codebook:       fp16   [n_clusters, max_blocksize]   # shared
self.row_to_int4:    int64  [out_features]   # -1 if row is binary
self.row_to_binary:  int64  [out_features]   # -1 if row is int4
self.int4_channels:  int64  [n_int4]   # which output rows are int4
self.binary_channels: int64 [n_binary] # which output rows are binary
self.bias:           fp16   [out_features] or None
```

Total bytes for a 3840×3840 layer with 5% int4 / 95% binary, blocksize 128:
- int4: 192 × 3840 = ~720 KB (int8) + ~400 B (scales) = ~720 KB
- binary bits: 3648 × 3840 / 8 = ~1.7 MB
- binary scales: 3648 × 30 = ~220 KB
- codebook_idx: 3648 × 30 = ~110 KB
- Total: **~2.7 MB per layer**

For 48 layers: **~130 MB quantized body**. Plus codebook (256 KB shared).
Plus BF16 embedding (~2 GB). Total: **~2.2 GB** to run the model.

Compare to BF16: 24 GB. **Compression ratio: ~11x for memory.**

## On-disk format

See `fabq_rc_cuda/src/fabq_rc_format.h` for the byte-exact layout. The
high-level structure:

- 32-byte header (magic `FBQB`, version, layer index, shapes)
- Int4 channels data
- Binary channels data (packed bits + scales + codebook indices)
- Optional bias (with NaN sentinel for "no bias")

The on-disk format is byte-identical to the in-memory representation after
`load_layer_from_file` — no unpacking, no transform. The kernel reads
exactly the bytes that are on disk.

## Streaming behavior

`build_bucket.py` uploads ~50 MB of stats + 256 KB of codebook + 48 × ~25
MB of pre-quantized shards to the HF bucket. The streaming notebook:

1. Downloads stats + codebook first (fast, ~1-2 sec)
2. Loads the BF16 model on GPU in BF16 (slow but only happens once,
   to get the tied embedding)
3. For each layer:
   a. Downloads the ~25 MB pre-quantized shard (1-2 sec on fast network)
   b. Calls `load_layer_from_file` to get the buffers
   c. Constructs a `QuantizedLinear` from the buffers
   d. Replaces the corresponding `nn.Linear` in the model
   e. Frees the shard from memory

**Cold start total:** 30-60 sec (mostly the BF16 model load for the
embedding). Per-shard download is overlapped with the previous layer's
quantization application, so the streaming overhead is mostly hidden.

**Peak VRAM during load:** ~26 GB (BF16 model + a few shards in flight).
**Peak VRAM after load:** ~2.2 GB (just the embedding + quantized body).

## Performance expectations (v1)

v1 is scalar — no tensor cores. Realistic throughput on A100 80GB:
- ~50-100 tokens/sec for 1-token generation
- ~1000-2000 tokens/sec for prefill with batch=1, seq=512

This is 5-10x slower than the BF16 baseline. v2 will add `mma.sync` /
`wmma` for the int4 submatrix (which should give most of the speed back
since int4 channels are 5% of params but contribute more than that to
the matmul cost).

## What's deferred to v2

- **Tensor cores for int4.** v1 is scalar.
- **Multi-tier codebook.** v1 uses one codebook for all layers; v2 uses
  4 tiers (Fisher-quartile-based) for a few % quality gain.
- **Kernel for the embedding lookup.** v1 uses the BF16 embedding via
  the standard PyTorch path.
- **Multimodal (vision + audio) encoders.** v1 is text-only.

## Trade-offs vs. "just use a standard quant"

| | Standard quant (Q4_K_M, etc.) | FABQ-RC v1 native | FABQ-RC v2 + tensor cores |
|---|---|---|---|
| Peak inference VRAM | 8 GB (Q4) | 2.2 GB | 2.2 GB |
| Per-token latency | fast (cuBLAS) | 5-10x slower (scalar) | ~2x slower than BF16 |
| Quality at same bpw | 4 bpw | 1.21 bpw | 1.21 bpw |
| Build complexity | low | high | high |
| First-run cost | none | 30-60 sec | 30-60 sec |

The v1 native-quantized inference is not a production-grade speed path
yet, but it is a real, working demonstration that you can run a 12B
model in 2.2 GB of VRAM with no FP16 weight materialization. v2 closes
the speed gap.

## Files in this folder

```
streaming/
├── fabq_rc_cuda/              # the C++/CUDA extension
│   ├── src/                   # C++/CUDA source
│   ├── quantized_linear.py    # the nn.Module
│   ├── model.py               # quantize_model_in_place
│   ├── io.py                  # file I/O wrappers
│   ├── kmeans.py              # shared codebook builder
│   ├── fisher.py              # Fisher pass
│   ├── quant_pipeline.py      # blocksize + allocation
│   ├── tests/test_kernel.py   # numerical correctness
│   └── setup.py
├── build_bucket.py            # one-time: populate the HF bucket
├── build_streaming_notebook.py
├── FABQ-RC-Gemma4-12B-Streaming.ipynb
└── STREAMING.md               # this file
```
