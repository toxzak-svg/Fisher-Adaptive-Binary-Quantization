# FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks

**Date:** 2026-04-05
**Status:** Active
**Author:** Zach Maronek + Marble

---

## Problem Statement

All existing 1-bit quantization methods use a fixed or semi-fixed blocksize across all layers. This is the wrong compromise. Weight distributions vary dramatically across layers — some layers are homogeneous (big blocks work fine), others are heterogeneous (need fine granularity). A single blocksize for all layers sacrifices quality everywhere.

**Goal:** Beat Q1_0_g128 (Bonsai's 1-bit format) and BiLLM on quality while staying at ~1.15-1.25 bits per parameter.

---

## Method Overview

FABQ-RC has four stages:

```
FP32 Weights
    │
    ▼
Stage 1: Fisher-Weighted Channel Importance
    │  Compute Fisher information per output channel
    │  Sort channels by importance
    ▼
Stage 2: Mixed-Precision Core Allocation
    │  Top 5% channels → int4 (preserve)
    │  Bottom 95% channels → binary ±1
    ▼
Stage 3: Adaptive Blocksize Selection
    │  Per-layer blocksize sweep {64, 128, 256, 512}
    │  Pick blocksize minimizing Fisher-weighted reconstruction error
    ▼
Stage 4: Residual Codebook Clustering
    │  Compute residuals: r = W - W_binary
    │  4 tiered codebooks × 64 centroids each (Fisher quartile-based)
    │  4-bit indices per block (16 centroids per layer cluster)
    ▼
FABQ-RC Quantized Model
```

**Effective bits per weight:** ~1.21 bpw

---

## Stage 1: Fisher-Weighted Channel Importance

### Why Fisher over Hessian/Magnitude?

Hessian = second derivative (curvature) — tells you loss curvature at the current point.
Fisher Information = expected gradient² — tells you, averaged over the data distribution, how much each parameter matters for the loss.

Fisher is more directly tied to the loss impact of quantizing a channel. We use it as the importance metric for channel allocation.

```python
# For a weight w_i, Fisher information:
# F_i = E[(∂ log p(y|x,θ) / ∂ w_i)²]
#
# Approximated as:
# F_i ≈ (1/N) Σ_n (∂L_n / ∂ w_i)²
#
# where gradients are from a calibration dataset
```

### Algorithm

```python
def compute_fisher_importance(model, calibration_loader, device):
    """
    Compute Fisher information per output channel for each linear layer.

    Returns: dict[layer_name, torch.Tensor of shape (out_channels,)]
    """
    model.eval()
    fisher = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module._fisher_grad = torch.zeros_like(module.weight)

    for batch in calibration_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.weight.grad is not None:
                    if module.weight.grad.dim() == 2:
                        grad_sq = module.weight.grad.data ** 2
                        channel_fisher = grad_sq.sum(dim=1)
                    else:
                        channel_fisher = grad_sq.sum(dim=(1, 2, 3))
                    module._fisher_grad += channel_fisher

        model.zero_grad()

    for name, module in model.named_modules():
        if hasattr(module, '_fisher_grad'):
            fisher[name] = module._fisher_grad / len(calibration_loader)

    return fisher
```

---

## Stage 2: Mixed-Precision Core Allocation

### The Insight

Most channels in a linear layer have low Fisher importance — their weights could be binarized with minimal impact on the loss. A small fraction of channels have very high Fisher importance — these are the "critical" channels that determine the layer's behavior.

We preserve only the most critical channels at int4. Everything else is binary.

**Why int4 instead of int8?** Going from int8 → int4 on the top 5% saves half those bits: `0.40 → 0.20`. At 5% of channels, quality degradation from int8→int4 is minimal because Fisher already identified these as the most important — they're dense and well-distributed.

### Algorithm

```python
def allocate_precision(fisher_scores, int4_fraction=0.05):
    """
    fisher_scores: list of (channel_idx, fisher_score) sorted descending
    int4_fraction: fraction of channels to preserve at int4 (default 5%)

    Returns: dict of {channel_idx: 'int4' or 'binary'}
    """
    n_int4 = max(1, int(len(fisher_scores) * int4_fraction))

    allocation = {}
    for i, (channel_idx, _) in enumerate(fisher_scores):
        if i < n_int4:
            allocation[channel_idx] = 'int4'
        else:
            allocation[channel_idx] = 'binary'

    return allocation
```

---

## Stage 3: Adaptive Blocksize Selection

### The Insight

Different layers have different weight distributions. A uniform-weight layer (e.g., embedding) can tolerate large blocks because all weights contribute equally. A heterogeneous layer (e.g., attention projection) needs smaller blocks to preserve important weight combinations.

Rather than pick one blocksize for the whole model, we pick the best blocksize **per layer** by optimizing reconstruction quality.

### Block Size Candidates

`B = {64, 128, 256, 512}` — 16 and 32 dropped due to scale overhead.

### Algorithm

```python
def compute_reconstruction_error(weights, block_size, fisher_channels):
    out_c, in_c = weights.shape
    errors = []

    for block_start in range(0, in_c, block_size):
        block_end = min(block_start + block_size, in_c)
        block_w = weights[:, block_start:block_end]

        block_fisher = fisher_channels.mean()

        scale = block_w.abs().mean() + 1e-8
        block_q = (block_w > 0).float() * 2 - 1
        block_recon = block_q * scale

        recon_error = ((block_w - block_recon) ** 2).mean()
        weighted_error = block_fisher * recon_error
        errors.append(weighted_error)

    return sum(errors)


def select_best_blocksize(weights, fisher_channels, candidates=[64, 128, 256, 512]):
    """Pick blocksize minimizing Fisher-weighted reconstruction error."""
    best_b, best_err = candidates[0], float('inf')

    for b in candidates:
        err = compute_reconstruction_error(weights, b, fisher_channels)
        if err < best_err:
            best_err = err
            best_b = b

    return best_b
```

---

## Stage 4: Residual Codebook

### The Insight

After binary quantization, systematic errors remain. The residual `r = W - Ŵ` for similar weight patterns is often similar — these are the "biases" introduced by forcing everything to ±scale. By clustering residual patterns, we can learn and correct for these systematic errors.

This is fundamentally different from BiLLM's "binary residual approximation," which approximates the residual as a linear function of the weight value. FABQ-RC uses a **discrete codebook** of residual patterns, which is non-linear and more expressive.

### Architectural Fix: Tiered Codebooks + 4-bit Indices

**Problem with global codebook:** Sharing a 256-centroid codebook across layers with very different Fisher profiles wastes most centroids.

**Fix:** Use **4 separate codebooks of 64 centroids each**, tiered by Fisher magnitude quartile. Same total storage (4 × 64 × 4 bytes = 8KB per blocksize), but much better residual coverage where it counts.

Additionally, use **4-bit indices** (16 centroids per layer cluster) instead of uint8 indices. This halves codebook overhead from 0.0625 bits/weight to ~0.03 bits/weight.

### Algorithm

```python
from sklearn.cluster import MiniBatchKMeans
import numpy as np


def build_tiered_codebooks(residual_blocks_by_fisher_quartile, n_clusters=64, seed=42):
    """
    residual_blocks_by_fisher_quartile: dict of {quartile: list of 2D numpy arrays}
    Each array has shape (block_out, block_size).
    n_clusters: centroids per codebook (64)

    Returns: dict of {quartile: codebook centroids}, each shape (64, block_size)
    """
    codebooks = {}

    for quartile, blocks in residual_blocks_by_fisher_quartile.items():
        if not blocks:
            codebooks[quartile] = np.zeros((n_clusters, blocks[0].shape[1]), dtype=np.float32)
            continue

        flat = np.array([b.flatten() for b in blocks])
        flat = flat.astype(np.float32)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1024)
        labels = kmeans.fit_predict(flat)
        centroids = kmeans.cluster_centers_

        codebooks[quartile] = centroids

    return codebooks
```

---

## Full Quantization Pipeline

```python
def quantize_fabq_rc(model, calibration_loader, device,
                     int4_fraction=0.05,
                     blocksize_candidates=[64, 128, 256, 512],
                     codebook_n_clusters=64):
    """
    Full FABQ-RC quantization pipeline.
    """
    fisher = compute_fisher_importance(model, calibration_loader, device)

    quantized_layers = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weights = module.weight.data.cpu().numpy()
        out_c, in_c = weights.shape

        f_scores = fisher.get(name, torch.ones(out_c))
        if isinstance(f_scores, torch.Tensor):
            f_scores = f_scores.cpu().numpy()

        sorted_channels = sorted(enumerate(f_scores), key=lambda x: x[1], reverse=True)

        allocation = allocate_precision(sorted_channels, int4_fraction)

        binary_channels = [i for i, a in allocation.items() if a == 'binary']
        int4_channels = [i for i, a in allocation.items() if a == 'int4']

        if binary_channels:
            binary_weights = weights[binary_channels, :]
            binary_fisher = f_scores[binary_channels]
            best_bs = select_best_blocksize(binary_weights, binary_fisher, blocksize_candidates)
        else:
            best_bs = 128

        quantized_layers[name] = QuantizedLayer(...)

    return quantized_layers
```

---

## Evaluation

### Perplexity

```python
def evaluate_perplexity(model, test_data, stride=512):
    """Compute perplexity on test data. Lower is better."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i in range(0, len(test_data) - 1, stride):
        input_ids = test_data[i:i+stride]
        target_ids = test_data[i+1:i+stride+1]

        with torch.no_grad():
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.view(-1, V), target_ids.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += target_ids.numel()

    return math.exp(total_loss / total_tokens)
```

### Benchmark Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Language Understanding | ARC-Easy/Challenge | Accuracy |
| Commonsense Reasoning | HellaSwag | Accuracy |
| Knowledge | TriviaQA | EM |
| Word-in-Context | WiC | Accuracy |
| Natural Questions | NQ | EM |

### Expected Results

| Method | bpw | Perplexity (7B) | Est. Quality Gap |
|--------|-----|-----------------|------------------|
| FP16 | 16.0 | baseline | — |
| Q1_0_g128 | 1.125 | poor | large |
| BiLLM | 1.08 | 8.41 (70B) | small |
| **FABQ-RC** | ~1.21 | target < 8.0 | smallest |

FABQ-RC should beat BiLLM because:
1. **Adaptive blocksize** recovers more per-layer quality than fixed blocksize
2. **Fisher > Hessian** as importance metric (direct loss relevance vs. curvature approximation)
3. **Residual codebook** corrects systematic binary quantization bias better than linear approximation

---

## Implementation Notes

- Calibration dataset: 512-1024 tokens from Pile subset (~10K samples)
- Fisher computation: requires one full forward+backward pass on calibration data
- k-means clustering: use `sklearn.MiniBatchKMeans` for efficiency
- **4 tiered codebooks** (64 centroids each) based on Fisher quartile — not a single global codebook
- **4-bit indices** per block (16 active centroids per layer cluster) — not uint8
- Memory overhead: codebook (4 × 64 × 128 × 4 bytes = 128KB) is negligible

## Out of Scope For v1

- Activation quantization (weight-only for now)
- Hardware-aware blocksize (GPU memory coalescing)
- Per-token vs per-block scale optimization
- Fine-tuning after quantization (QAT)

---

## Changelog

### v1 → v2

1. **int8 → int4 for top 5% channels**: saves 0.20 bits/weight
2. **Dropped blocksize 16, 32**: minimum blocksize 64 to avoid scale overhead inflation
3. **Single global codebook → 4 tiered codebooks (64 centroids each)**: better residual coverage per Fisher tier
4. **uint8 indices → 4-bit indices**: 16 centroids per layer cluster, halves codebook overhead
5. **Corrected bpw math**: Total now ~1.21 bpw

---

## License

Apache 2.0
