# FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks

**Method + Kaggle benchmark notebook**

---

## What Is This?

FABQ-RC is a new 1-bit quantization method for large language models that adapts per layer rather than using a fixed blocksize. It was designed to beat Q1_0_g128 (Bonsai's format) and BiLLM on quality while staying at ~1.15-1.20 bits per parameter.

---

## Files

```
fabq_rc/
├── FABQ_RC_SPEC.md    ← Full method specification (read this first)
├── FABQ_RC.ipynb       ← Kaggle notebook (upload to Kaggle to run)
└── README.md          ← This file
```

---

## Key Ideas

### The Problem with Fixed Blocksize

All existing 1-bit methods (Q1_0_g128, BiLLM) use the same blocksize for every layer. But weight distributions vary — a layer with uniform weights can tolerate 256-wide blocks, while a heterogeneous layer needs 16-wide blocks to preserve important combinations. A single blocksize is always the wrong compromise for some layers.

### FABQ-RC's Four Stages

1. **Fisher-Weighted Channel Importance** — Instead of magnitude or Hessian, use Fisher Information per output channel to determine which weights actually matter for the loss
2. **Mixed-Precision Core** — Top 5% of channels by Fisher → int8 (accurate). Remaining 95% → binary ±1 (compact)
3. **Adaptive Blocksize** — Sweep {16, 32, 64, 128, 256} per layer, pick the one minimizing Fisher-weighted reconstruction error
4. **Residual Codebook** — After binary quantization, cluster the systematic residual errors into a k-means codebook (256 centroids). During inference, add the centroid back. This corrects the systematic bias that binary quantization introduces.

### Why Fisher > Hessian

Hessian = second derivative (curvature) — tells you loss curvature at the current point.
Fisher Information = expected gradient² — tells you, averaged over the data distribution, how much each parameter matters.

Fisher is more directly tied to the loss impact of quantizing a channel. We use it as the importance metric for channel allocation.

### Why Residual Codebook > Linear Approximation

BiLLM approximates the residual as a linear function of the weight value. This misses nonlinear systematic errors. FABQ-RC's k-means codebook captures arbitrary residual patterns, which is more expressive and doesn't assume a functional form.

---

## Running the Notebook on Kaggle

1. Upload `FABQ_RC.ipynb` to Kaggle
2. Add input dataset: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (or let it auto-download)
3. Set accelerator: **GPU P100**
4. Set internet: **On**
5. Run all cells

**Runtime:** ~30-45 min on P100

---

## Quick Method Reference

| Stage | What it does |
|-------|-------------|
| Fisher importance | Per-channel importance scores via gradient² proxy |
| Precision allocation | Top 5% channels → int8, rest → binary |
| Adaptive blocksize | Per-layer blocksize selection by reconstruction error |
| Residual codebook | k-means on quantization residuals, 256 centroids, shared across layers |

**Effective bits:** ~1.15-1.20 bpw (vs Q1_0_g128's 1.125 bpw at equal or better quality)

---

## Expected Results

| Method | bpw | Perplexity | Notes |
|--------|-----|------------|-------|
| FP16 | 16.0 | baseline | |
| Q1_0_g128 | 1.125 | degraded | Bonsai's format |
| BiLLM | 1.08 | ~8.41 (70B) | Best prior work |
| **FABQ-RC** | ~1.18 | target < 8.0 | Our method |

---

## Method Origin

Designed by Zach Maronek, 2026-04-05. The core insight — that per-layer adaptive blocksize is the biggest untapped lever in 1-bit quantization — came from analyzing why Q1_0_g128 degrades badly on large models and why BiLLM's fixed blocksize still leaves quality on the table.
