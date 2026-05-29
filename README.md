# FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks

## A Vigorous Scientific Research Experiment

**Status:** Active
**Duration:** April 2026 - Present

---

## What Is FABQ-RC?

FABQ-RC is a 1-bit quantization method for large language models that adapts per layer rather than using a fixed blocksize. It combines:

1. **Fisher-Weighted Channel Importance** — Which channels actually matter for loss?
2. **Mixed-Precision Core Allocation** — int8 for critical channels, binary for the rest
3. **Adaptive Blocksize** — Per-layer blocksize selection, not global
4. **Residual Codebook** — k-means corrects systematic binary bias

**Target:** ~1.18 bits per parameter, beating BiLLM on quality

---

## The Method

### Why Fisher > Hessian > Magnitude

| Metric | What it measures | Problem |
|--------|-----------------|---------|
| **Magnitude** | Weight absolute value | Big weights aren't always important |
| **Hessian** | Loss curvature at current θ | Local only, expensive to compute |
| **Fisher** | Expected gradient² over data | Captures average importance, tractable |

### Four Stages

```
FP32 Weights
    │
    ▼
Stage 1: Fisher-Weighted Channel Importance
    │
Stage 2: Mixed-Precision Core Allocation
    │  Top 5% channels → int4
    │  Bottom 95% channels → binary ±1
    ▼
Stage 3: Adaptive Blocksize Selection
    │  Per-layer blocksize {64, 128, 256, 512}
    ▼
Stage 4: Residual Codebook Clustering
    │  4 tiered codebooks × 64 centroids
    │  4-bit indices per block
    ▼
FABQ-RC Quantized Model
    │
    ▼
GGUF Export
```

### Why Residual Codebook > Linear Approximation

BiLLM approximates residuals as a linear function of the weight value. FABQ-RC's k-means codebook is nonlinear and captures arbitrary residual patterns without assuming a functional form.

---

## Quick Start

### Download the Model

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download("toxzak/Qwen3.6-27B-FABQ-RC-GGUF")
```

### Use with llama.cpp

```bash
# Example inference command
./llama-cli -m Qwen3.6-27B-FABQ-RC-Q4_K_M.gguf -n 256 -p "The future of 1-bit quantization is"
```

### Evaluate

```python
# Perplexity on WikiText-2
./llama-perplexity -m Qwen3.6-27B-FABQ-RC-Q4_K_M.gguf -f wikitext.txt
```

---

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen/Qwen3.6-27B |
| **Format** | GGUF |
| **Bits per parameter** | ~1.18 bpw |
| **Architecture** | FABQ-RC (Fisher-Adaptive Binary Quantization with Residual Codebooks) |
| **Calibration** | C4 dataset, 2048 samples |

---

## Key Results

| Method | bpw | Perplexity | Notes |
|--------|-----|------------|-------|
| FP16 | 16.0 | baseline | |
| Q1_0_g128 | 1.125 | degraded | Bonsai's format |
| BiLLM | 1.08 | ~8.41 (70B) | Best prior work |
| **FABQ-RC** | ~1.18 | target < 8.0 | Our method |

---

## Files

```
fabq-rc/
├── README.md                              ← This file
├── FABQ_RC_SPEC.md                       ← Full method specification
├── FABQRC_PLAN.md                        ← Research plan
├── Main-FABQ-RC-Notebook.ipynb          ← Main quantization notebook
├── FABQ-RC-Dense-27B-Notebook.ipynb     ← Dense model experiments
└── plans/
    ├── CALIBRATION-ROBUSTNESS-PLAN.md  ← Calibration improvements
    ├── FABQ-VP-SPEC.md                  ← Variable precision extension
    ├── EBQ-SPEC.md                      ← Error-budget allocation
    └── UNIFIED-SPEC.md                   ← Combined architecture
```

---

## Citation

```
FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks
Zach Maronek, 2026
```

---

## License

Apache 2.0 (see Hugging Face model page for details)

---

*Built by Zach Maronek · April 2026*
