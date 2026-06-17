# FABQ-VP: Variable Precision Extension for FABQ-RC

**Date:** 2026-05-04
**Status:** Planned
**Parent:** FABQ_RC_SPEC.md
**Goal:** Extend FABQ-RC's adaptive precision concepts from 1-bit to 2-8 bits per parameter

---

## Motivation

FABQ-RC proves that adaptive per-layer precision allocation beats fixed blocksize. FABQ-VP generalizes this to variable precision across a continuous range [1, 8] bits.

---

## 1. Motivation

FABQ-RC proves that adaptive per-layer precision allocation beats fixed blocksize. Currently FABQ-RC works at exactly 1-bit (binary ±1 + int8). FABQ-VP generalizes this to variable precision across a continuous range [1, 8] bits.

### Why Extend Beyond 1-bit?

1. **Different model families tolerate different bit-widths** — Qwen 35B responds well to 4-bit schemes (AWQ, GPTQ), suggesting headroom between 1-bit and full 16-bit
2. **Per-channel allocation is still coarse at 1-bit** — FABQ-RC splits channels into int8 vs binary, but many channels could be 3-4 bits with minimal size cost and significant quality gain
3. **Error-budget quantization needs intermediate precision** — the global knapsack allocator works best with fine-grained bit-width choices

---

## 2. Core Concept: Fisher-Weighted Variable Precision Allocation

### 2.1 From Binary to N-bit

FABQ-RC allocates: `int8` (top 5%) vs `binary ±1` (rest).  
FABQ-VP allocates: `fp16` (top 0.5%) vs `int8` (next 4.5%) vs `int4` (next 20%) vs `int2` (next 25%) vs `binary ±1` (rest).

This creates a **precision pyramid** where Fisher importance directly maps to bit-width.

### 2.2 Algorithm

```python
def allocate_variable_precision(fisher_scores, weight_tensor, pyramid={
    'fp16': 0.005,    # 0.5% of channels
    'int8': 0.045,    # 4.5% of channels  
    'int4': 0.200,    # 20% of channels
    'int2': 0.250,    # 25% of channels
    'binary': 0.500   # 50% of channels
}):
    """
    fisher_scores: per-channel importance (gradient² proxy)
    weight_tensor: original fp16 weights for calibration
    pyramid: cumulative fraction allocated to each precision level
    
    Returns: dict mapping channel idx → precision level
    """
    # Sort channels by Fisher importance (descending)
    sorted_indices = torch.argsort(fisher_scores, descending=True)
    n_channels = len(fisher_scores)
    
    allocation = {}
    cumulative = 0
    
    for precision, fraction in pyramid.items():
        threshold = int(n_channels * fraction)
        for i in range(cumulative, min(cumulative + threshold, n_channels)):
            allocation[sorted_indices[i].item()] = precision
        cumulative += threshold
    
    return allocation
```

### 2.3 Per-Layer Blocksize for Non-Binary Precision

FABQ-RC's adaptive blocksize applies to binary weights. For int4/int8 weights in FABQ-VP:

- **int8**: per-channel scales (no blocksize needed — 8 bits per value anyway)
- **int4**: per-group blocksize {16, 32, 64} selected by reconstruction error
- **int2**: per-group blocksize {32, 64, 128} selected by reconstruction error
- **binary**: keep FABQ-RC's blocksize search {16, 32, 64, 128, 256}

---

## 3. Residual Codebook Extension

FABQ-RC's residual codebook clusters `W - W_binary` (1-bit residuals).  
FABQ-VP extends this to multi-level residuals:

```python
def build_multi_level_residual_codebook(layers, precision_allocation):
    """
    layers: dict of {layer_name: fp16_weights}
    precision_allocation: dict of {layer_name: {channel_idx: precision}}
    
    For each precision level, build a separate codebook:
    - codebook_int4: clusters W_int8 - W_int4 residuals
    - codebook_int2: clusters W_int4 - W_int2 residuals  
    - codebook_binary: clusters W_int2 - W_binary residuals (FABQ-RC style)
    
    Returns: dict of codebooks per precision transition
    """
    residual_codebooks = {}
    
    # Build codebooks for each precision transition
    for precision_pair in [('int4', 'int8'), ('int2', 'int4'), ('binary', 'int2'):
        lower_prec, upper_prec = precision_pair
        
        # Collect residuals at this level across all layers
        residuals = []
        for layer_name, weights in layers.items():
            alloc = precision_allocation[layer_name]
            
            # Get weights at upper precision (more bits)
            upper_weights = quantize_to_precision(weights, upper_prec)
            # Get weights at lower precision (fewer bits)
            lower_weights = quantize_to_precision(weights, lower_prec)
            
            # Residual = what we lose going from upper to lower precision
            residual = upper_weights - lower_weights
            
            # Sample blocks for clustering
            for block in iterate_blocks(residual, blocksize=128):
                residuals.append(block.flatten())
        
        # Cluster into 256 centroids
        codebook = kmeans_cluster(residuals, n_clusters=256)
        residual_codebooks[precision_pair] = codebook
    
    return residual_codebooks
```

---

## 4. Memory Budget Analysis

### 4.1 Size Target for Qwen 35B

| Precision | Fraction | Bits/Param | Size (35B) |
|-----------|----------|------------|------------|
| fp16 | 0.5% | 16.0 | 0.28 GB |
| int8 | 4.5% | 8.0 | 1.26 GB |
| int4 | 20% | 4.0 | 2.80 GB |
| int2 | 25% | 2.0 | 1.75 GB |
| binary | 50% | 1.0 | 1.75 GB |
| **Total** | 100% | **~3.84 bpw** | **~16.8 GB** |

This fits easily in RAM (no VRAM needed for weights).

### 4.2 Alternative: 3-bit Average Target

If target is 3 bpw for 35B (total 13 GB):

| Precision | Fraction | Bits/Param | Size (35B) |
|-----------|----------|------------|------------|
| int8 | 10% | 8.0 | 2.80 GB |
| int4 | 40% | 4.0 | 5.60 GB |
| int2 | 35% | 2.0 | 2.45 GB |
| binary | 15% | 1.0 | 0.53 GB |
| **Total** | 100% | **~3.85 bpw** | **~13.4 GB** |

---

## 5. Inference Architecture

### 5.1 Weight Loading Strategy

```
                    ┌─────────────────────────────────────────────┐
                    │              Host RAM                        │
                    │  ┌────────────────────────────────────────┐  │
                    │  │  Variable-precision weight storage    │  │
                    │  │  - fp16 islands (0.3 GB)               │  │
                    │  │  - int8 blocks (1.3 GB)                │  │
                    │  │  - int4 blocks (2.8 GB)                │  │
                    │  │  - int2 blocks (1.8 GB)                │  │
                    │  │  - binary blocks (1.8 GB)              │  │
                    │  └────────────────────────────────────────┘  │
                    │                     │                        │
                    │              streaming dequantization        │
                    └─────────────────────│────────────────────────┘
                                        │  int8/FP16 activations
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │         CPU Compute (or tiny GPU)            │
                    │  ┌────────────────────────────────────────┐  │
                    │  │  Quantized matmul kernels              │  │
                    │  │  - int8 × int8 → int32 accumulation     │  │
                    │  │  - int4 × int4 → int16 accumulation     │  │
                    │  │  - mixed precision dispatch             │  │
                    │  └────────────────────────────────────────┘  │
                    │                     │                        │
                    └─────────────────────│────────────────────────┘
```

### 5.2 KV-Cache Quantization

KV cache dominates memory at long context lengths. FABQ-VP includes:

```python
def quantize_kv_cache(k_cache, v_cache, target_bits=4):
    """
    k_cache: (batch, heads, seq_len, head_dim) in FP16
    v_cache: (batch, heads, seq_len, head_dim) in FP16
    
    Returns: (quantized_k, quantized_v, scales) with ~4 bits per value
    """
    # Per-head quantization for K (more sensitive)
    k_scales = k_cache.abs().amax(dim=-1, keepdim=True) / 7.5  # int4 range
    k_quant = torch.clamp(torch.round(k_cache / k_scales), -8, 7).to(torch.int8)
    
    # Per-head quantization for V (less sensitive, can use coarser)
    v_scales = v_cache.abs().amax(dim=-1, keepdim=True) / 7.5
    v_quant = torch.clamp(torch.round(v_cache / v_scales), -8, 7).to(torch.int8)
    
    return k_quant, v_quant, (k_scales, v_scales)
```

---

## 6. Relationship to FABQ-RC

| Aspect | FABQ-RC | FABQ-VP |
|--------|---------|---------|
| Precision levels | 2 (int8, binary) | 5 (fp16, int8, int4, int2, binary) |
| Bit range | 1.0-1.2 bpw | 2.0-4.0 bpw (configurable) |
| Per-layer blocksize | Yes, {16-256} for binary | Yes, per precision level |
| Residual codebook | 1 level | 3 levels (one per transition) |
| Fisher importance | Channel-level | Channel-level (unchanged) |
| Calibration data | Same | Same |
| Model targets | TinyLlama, Mistral 7B | Qwen 35B, Llama 70B |

---

## 7. Implementation Plan

### Phase A: FABQ-RC Validation (Current)

1. Fix padded-block centroid bug in FABQ-RC
2. Validate perplexity improvements
3. Establish baseline numbers

### Phase B: FABQ-VP Prototype

1. Extend precision allocation to 5 levels
2. Implement per-precision residual codebooks
3. Test on TinyLlama first (fast iteration)
4. Profile memory and compute

### Phase C: Qwen 35B Integration

1. Port FABQ-VP to Qwen architecture
2. Calibrate on domain-appropriate data
3. Evaluate on WikiText2 and downstream tasks

---

## 8. Open Questions

1. **Optimal pyramid fractions** — should be learned from validation perplexity, not hand-tuned
2. **Cross-layer codebook sharing** — does a single int4→int8 codebook work across all layers?
3. **Activation quantization** — weight-only for now, activations are harder to calibrate
4. **AWQ-style salient weights** — FABQ-RC's int8 channels could be further split into salient (FP16) vs non-salient (int8)