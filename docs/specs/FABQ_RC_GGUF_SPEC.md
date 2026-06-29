# FABQ-RC GGUF Integration Specification

## Overview

FABQ-RC (Fisher-Adaptive Binary Quantization with Residual Codebooks) requires custom GGUF tensor support in llama.cpp because:
1. Adaptive per-layer blocksizes {64, 128, 256, 512}
2. Mixed int4/binary precision per channel
3. Residual codebook with 4 tiers × 64 centroids

## GGUF Type Assignment

Using `GGML_TYPE_FABQ_RC = 41` (after Q1_0).

## Tensor Data Format

### Per-Layer GGUF Metadata (kv pairs)
```
fabq.version = u32 (currently 1)
fabq.layer.{n}.blocksize = u32 {64, 128, 256, 512}
fabq.layer.{n}.int4_channels = u32 (count of int4 channels)
fabq.layer.{n}.binary_channels = u32 (count of binary channels)
fabq.layer.{n}.fisher_quartile = u32 (0-3, selects codebook tier)
```

### Tensor Data Structure for block_q_fabq_rc

```
struct block_fabq_rc {
    // Block header (8 bytes)
    uint16_t blocksize;      // 64, 128, 256, or 512
    uint16_t n_int4;          // number of int4 elements in this block
    uint16_t n_binary;        // number of binary elements in this block
    uint16_t flags;           // reserved

    // Per-block scales (8 bytes)
    fp16_t binary_scale;      // scale for binary weights
    fp16_t int4_scale;        // scale for int4 weights
    fp16_t binary_min;        // minimum for binary
    fp16_t int4_min;          // minimum for int4

    // int4 weights (n_int4/2 bytes, nibble packed)
    // Format: 2 elements per byte, MSB first = element[i], LSB = element[i+1]

    // Binary weights (n_binary bits, packed into bytes)
    // Format: 1 bit per weight, packed row-major

    // Codebook index (4 bits per block for residual correction)
    // Points to one of 16 centroids in the tier's codebook
};
```

### Codebook Storage

4 tiered codebooks stored as separate tensors:
- `fabq.codebook.0` : float32[64, blocksize]  - Fisher quartile 0 (lowest)
- `fabq.codebook.1` : float32[64, blocksize]  - Fisher quartile 1
- `fabq.codebook.2` : float32[64, blocksize]  - Fisher quartile 2
- `fabq.codebook.3` : float32[64, blocksize]  - Fisher quartile 3 (highest)

## Dequantization Algorithm

```c
void dequantize_fabq_rc(const block_fabq_rc * blocks, float * output, int64_t n) {
    for each block:
        // Reconstruct binary weights
        for i in 0..n_binary:
            bit = (block->binary_data[byte_idx] >> bit_idx) & 1;
            output[i] = (bit ? 1 : -1) * block->binary_scale + block->binary_min;

        // Reconstruct int4 weights
        for i in 0..n_int4:
            nibble = (block->int4_data[byte_idx] >> 4) if i%2==0 else (block->int4_data[byte_idx] & 0x0F);
            output[i + n_binary] = nibble * block->int4_scale + block->int4_min;

        // Apply residual correction from codebook
        centroid_idx = block->codebook_index;
        for i in 0..blocksize:
            output[i] += codebook[fisher_quartile][centroid_idx][i];
}
```

## Implementation Tasks

1. Add `GGML_TYPE_FABQ_RC = 41` to `ggml/include/ggml.h`
2. Define `struct block_fabq_rc` in `ggml-quants.h`
3. Add `dequantize_row_fabq_rc()` function declaration
4. Implement `dequantize_row_fabq_rc()` in `ggml-quants.c`
5. Add FABQ-RC to type switch statements in all backends (CPU, CUDA, Vulkan, Metal, etc.)
6. Add model loading support in `llama-model-loader.cpp`
7. Handle codebook tensor loading and validation

## File Changes Required

### ggml/include/ggml.h
- Add `GGML_TYPE_FABQ_RC = 41` to enum

### ggml/src/ggml-quants.h
- Add `block_fabq_rc` struct definition
- Add `dequantize_row_fabq_rc()` declaration

### ggml/src/ggml-quants.c
- Implement `dequantize_row_fabq_rc()` with SIMD optimization

### ggml/src/ggml-backend-cpu.c
- Add FABQ-RC case to dequantization switch

### Other backends
- CUDA, Vulkan, Metal, SYCL, OpenCL, OpenVINO, WebGPU, etc.

### src/llama-model-loader.cpp
- Parse FABQ-RC tensor metadata
- Validate codebook tensors
- Set up tensor views for FABQ-RC blocks