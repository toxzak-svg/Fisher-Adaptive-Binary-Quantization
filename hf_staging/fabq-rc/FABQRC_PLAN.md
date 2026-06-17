# FABQ-RC Research Plan

## Status: Active

---

## Current State

- **Core method:** 1-bit quantization with adaptive per-layer blocksize + residual codebook
- **Implementation:** Working notebook with Qwen3.6-27B
- **Model published:** toxzak/Qwen3.6-27B-FABQ-RC-GGUF on Hugging Face
- **Format:** GGUF export

---

## Phase 1: Validation

### 1.1 Verify compressed model size
- Confirm actual bpw matches theoretical (~1.18)
- Measure file size after GGUF export

### 1.2 Perplexity benchmarks
- WikiText-2 test set
- Compare against FP16 baseline
- Compare against Q1_0_g128 and BiLLM at same bpw

**Status:** Model published, validation pending

---

## Phase 2: Ablation Studies

### 2.1 Codebook architecture
- **Original:** 4 tiered codebooks (64 centroids each), 4-bit indices
- **Ablate:** Single global codebook, uint8 indices
- Measure quality vs compression tradeoff

### 2.2 int4 vs int8 for top channels
- **Original:** int4 for top 5% channels
- **Ablate:** int8 for top 5% channels
- Measure quality degradation

### 2.3 Blocksize search range
- Candidates: {64, 128, 256, 512}
- Check optimal ceiling by model size

### 2.4 Codebook index bit-width
- 4-bit (16 centroids per cluster) vs uint8 (256 centroids)

### 2.5 Fisher vs Hessian vs Magnitude
- Compare allocation strategies

---

## Phase 3: Baseline Comparisons

### 3.1 Run against available baselines
- Q1_0_g128
- BiLLM
- QuIP#2, AQLM (if available)

### 3.2 Standardized comparison
- Same model architecture
- Same benchmark dataset
- Same bpw across methods

---

## Phase 4: Scale Testing

### 4.1 Smaller models (confirm advantage holds)
- Test on 7B scale
- Compare against Q1_0_g128 and BiLLM at same bpw

### 4.2 Medium models (13B, 14B)
- Same benchmark suite
- Test if advantage scales

### 4.3 Larger models (70B+)
- Check per-layer adaptive blocksize importance at scale

---

## Phase 5: Extensions

### 5.1 FABQ-VP: Variable Precision
- Extend from 2 levels (int4, binary) to 5 (fp16, int8, int4, int2, binary)
- Target ~3-4 bpw for Qwen 35B

### 5.2 EBQ: Error-Budget Quantization
- Global perplexity-constrained bit allocation
- RAM-only inference for 35B models

### 5.3 KV Cache Quantization
- Profile sensitivity at different sequence lengths
- Add quantized KV cache support

---

## Calibration Robustness

> **Known concern:** Fisher scores depend on calibration distribution. Single-domain calibration (C4) may produce domain-specific importance rankings.

Potential improvements:
- Multi-domain Fisher aggregation (C4 + code + math + long-context)
- Domain-spike detection for channels important in specific domains
- Bootstrap confidence intervals for stability

---

## Immediate Next Steps

1. **Validate model quality** — Run perplexity benchmarks on quantized model
2. **Confirm compression ratio** — Verify actual bpw matches theoretical
3. **Compare with baselines** — Q1_0_g128, BiLLM at same bpw

---

*FABQRC_PLAN.md — updated 2026-05-28*
