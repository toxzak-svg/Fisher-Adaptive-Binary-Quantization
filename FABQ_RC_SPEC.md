# FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks

**Date:** 2026-04-05
**Status:** Draft v2 (bpw math corrected)
**Author:** Zach Maronek + Marble

---

## Problem Statement

All existing 1-bit quantization methods use a fixed or semi-fixed blocksize across all layers. This is the wrong compromise. Weight distributions vary dramatically across layers — some layers are homogeneous (big blocks work fine), others are heterogeneous (need fine granularity). A single blocksize for all layers sacrifices quality everywhere.

**Goal:** Beat Q1_0_g128 (Bonsai's 1-bit format) and BiLLM on quality while staying at ~1.15-1.25 bits per parameter (corrected target range).

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
    │  (16, 32 dropped — small blocks inflate scale overhead)
    │  Pick blocksize minimizing Fisher-weighted reconstruction error
    ▼
Stage 4: Residual Codebook Clustering
    │  Compute residuals: r = W - W_binary
    │  4 tiered codebooks × 64 centroids each (Fisher quartile-based)
    │  4-bit indices per block (16 centroids per layer cluster)
    ▼
FABQ-RC Quantized Model
```

**Effective bits per weight (corrected):** ~1.21 bpw

| Component | Original (flawed) | Corrected |
|-----------|-------------------|-----------|
| int4 channels (5%) | 0.40 (int8) | 0.20 |
| Binary channels (95%) | 0.95 | 0.95 |
| Scale factors | 0.02–0.12 (variable) | ~0.03 (min bs=64) |
| Codebook indices | 0.06 (256 global) | 0.03 (4-bit tiered) |
| **Total** | **~1.38–1.53** | **~1.21** |

The original spec claimed "~1.38 bpw at 5% int8 → 1.20 at 2% int8" but the codebook overhead compounds per block (uint8 index per 128 weights = 0.0625 bits/weight), which was treated as negligible but isn't. With the fixes below, we genuinely reach ~1.21 bpw.

---

## Stage 1: Fisher-Weighted Channel Importance

### Why Fisher over Hessian/Magnitude?

Weight magnitude tells you how big a weight is. Hessian tells you how much the loss changes when the weight changes. Fisher information tells you *on average* how much the loss changes — averaged over the data distribution.

```python
# For a weight w_i, Fisher information:
# F_i = E[(∂ log p(y|x,θ) / ∂ w_i)²]
#
# Approximated as:
# F_i ≈ (1/N) Σ_n (∂L_n / ∂ w_i)²
#
# where gradients are from a calibration dataset

# Practical approximation: use activations as proxy
# For output channel j in layer l:
# F_j = (1/|C_j|) Σ_{k∈C_j} E[∂L/∂w_k]²
#      ≈ (1/|C_j|) Σ_{k∈C_j} (â_k · ĥ_k)²
# where â_k is the gradient w.r.t. activation and ĥ_k is the activation
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

    # Accumulate gradients squared per channel
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Hook to capture gradients during backprop on calibration data
            module._fisher_grad = torch.zeros_like(module.weight)

    # Run forward + backward on calibration data
    for batch in calibration_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()

        # Accumulate gradient² per output channel
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.weight.grad is not None:
                    # Sum over input dimension to get per-output-channel Fisher
                    if module.weight.grad.dim() == 2:
                        grad_sq = module.weight.grad.data ** 2
                        # Per output channel: sum over input dim
                        channel_fisher = grad_sq.sum(dim=1)  # (out_channels,)
                    else:
                        channel_fisher = grad_sq.sum(dim=(1, 2, 3))  # conv
                    module._fisher_grad += channel_fisher

        model.zero_grad()

    # Normalize
    for name, module in model.named_modules():
        if hasattr(module, '_fisher_grad'):
            fisher[name] = module._fisher_grad / len(calibration_loader)

    return fisher
```

### Channel Sorting

```python
def sort_channels_by_fisher(fisher, layer_name, module):
    """
    Sort output channels by Fisher importance (descending).
    Returns: list of (channel_idx, fisher_score) sorted by importance
    """
    f = fisher[layer_name]
    order = torch.argsort(f, descending=True)
    return [(int(idx), float(f[idx])) for idx in order]
```

---

## Stage 2: Mixed-Precision Core Allocation

### The Insight

Most channels in a linear layer have low Fisher importance — their weights could be binarized with minimal impact on the loss. A small fraction of channels have very high Fisher importance — these are the "critical" channels that determine the layer's behavior.

We preserve only the most critical channels at int4. Everything else is binary.

**Why int4 instead of int8?** Going from int8 → int4 on the top 5% saves half those bits: `0.40 → 0.20`. At 5% of channels, quality degradation from int8→int4 is minimal because Fisher already identified these as the most important — they're dense and well-distributed. This single change gets us from ~1.38 bpw to ~1.18 bpw.

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

### Mixed-Precision Weight Representation

```python
# For a linear layer with shape (out_channels, in_channels):
#
# int4 channels: store as int4 + per-channel FP16 scale
# binary channels: store as bit vector + per-block FP16 scale
#
# Layout in memory:
# ┌─────────────────────────────────────────────────────────────┐
# │ int4_weights[ni, ki]  │  binary_weights_bitvec[nb]         │
# │ int4_scales[ni]        │  binary_block_scales[nb_blocks]     │
# │ channel_allocation[out_channels]  (1=int4, 0=binary)       │
# │ int4_channel_indices[ni]                                  │
# └─────────────────────────────────────────────────────────────┘
```

---

## Stage 3: Adaptive Blocksize Selection

### The Insight

Different layers have different weight distributions. A uniform-weight layer (e.g., embedding) can tolerate large blocks because all weights contribute equally. A heterogeneous layer (e.g., attention projection) needs smaller blocks to preserve important weight combinations.

Rather than pick one blocksize for the whole model, we pick the best blocksize **per layer** by optimizing reconstruction quality.

### Block Size Candidates

`B = {64, 128, 256, 512}` — **16 and 32 dropped**

Why drop 16 and 32? Small blocksizes dramatically inflate the number of scale values stored. A blocksize-16 layer with 4096 output channels stores 256 scales per output channel vs. 32 for blocksize-128. Since scales are FP16, this isn't "0.02 bits" anymore — at blocksize 16 it's closer to 0.125 bits/weight. Enforcing a minimum blocksize of 64 recovers ~0.05–0.10 bpw.

For each layer, for each candidate blocksize `b ∈ B`:
1. Partition weights into blocks of size `b` (last block may be smaller)
2. For each block, compute the quantization error weighted by Fisher importance of channels in that block
3. Sum weighted errors across all blocks
4. Pick `b` minimizing the weighted sum

```python
def compute_reconstruction_error(weights, block_size, fisher_channels):
    """
    weights: (out_channels, in_channels)
    fisher_channels: (out_channels,) — Fisher importance per output channel
    """
    out_c, in_c = weights.shape
    errors = []

    for block_start in range(0, in_c, block_size):
        block_end = min(block_start + block_size, in_c)
        block_w = weights[:, block_start:block_end]  # (out_c, block_size)

        # Fisher-weighted block importance
        block_fisher = fisher_channels.mean()  # average over the block's output channels

        # Quantize block to ±1
        scale = block_w.abs().mean() + 1e-8
        block_q = (block_w > 0).float() * 2 - 1  # ±1
        block_recon = block_q * scale

        # Reconstruction error
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

**Problem with original design:** Sharing a 256-centroid codebook across layers with very different Fisher profiles wastes most centroids. A single global codebook can't cover residual variance well across all layers.

**Fix:** Use **4 separate codebooks of 64 centroids each**, tiered by Fisher magnitude quartile. Same total storage (4 × 64 × 4 bytes = 8KB per blocksize), but much better residual coverage where it counts because residuals within a Fisher tier have tighter structure.

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

        # Flatten blocks for k-means
        flat = np.array([b.flatten() for b in blocks])  # (n_blocks, block_size)
        flat = flat.astype(np.float32)

        # Cluster residual patterns
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1024)
        labels = kmeans.fit_predict(flat)
        centroids = kmeans.cluster_centers_  # (n_clusters, block_size)

        codebooks[quartile] = centroids

    return codebooks


def apply_residual_correction(weights, block_size, codebook, n_centroids=16):
    """
    Quantize weights + apply residual codebook correction.
    Uses 4-bit indices (n_centroids=16 per layer cluster).

    Returns: (quantized_weights, scale, codebook_indices)
    """
    out_c, in_c = weights.shape
    quantized = np.zeros_like(weights)
    indices = []

    for block_start in range(0, in_c, block_size):
        block_end = min(block_start + block_size, in_c)
        block_w = weights[:, block_start:block_end]

        # Binary quantization
        scale = block_w.std() + 1e-8
        block_q = np.where(block_w > 0, 1.0, -1.0)

        # Find nearest residual centroid (among 16 active centroids)
        residual = block_w - block_q * scale  # (out_c, block_size)
        res_flat = residual.flatten().reshape(1, -1)
        centroid_idx = ((codebook[:n_centroids] - res_flat) ** 2).sum(axis=1).argmin()

        # Apply correction
        corrected = block_q * scale + codebook[centroid_idx]
        quantized[:, block_start:block_end] = corrected
        indices.append(centroid_idx)

    return quantized, indices
```

### Memory Layout

```
Per layer:
  - binary_weights_bitvec[nb_blocks, block_size]  → 1 bit/weight
  - binary_block_scales[nb_blocks]                 → FP16
  - codebook_indices[nb_blocks]                    → 4 bits/index (16 centroids)
  - int4_channel_mask[out_channels]                 → 1 bit/channel
  - int4_scales[ni]                                → FP16
  - int4_weights[ni, ki]                           → 4 bits/weight

Per model:
  - 4 tiered codebooks × 64 centroids × blocksize × 4 bytes
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

    Returns: dict of {layer_name: QuantizedLayer} with:
        - int4_weights, int4_scales, int4_channel_indices
        - binary_weights_bitvec, binary_block_scales, binary_blocksize
        - codebook (tiered), codebook_indices
        - allocation (per-channel precision assignment)
    """
    # Stage 1: Fisher importance
    fisher = compute_fisher_importance(model, calibration_loader, device)

    quantized_layers = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weights = module.weight.data.cpu().numpy()
        out_c, in_c = weights.shape

        # Get Fisher scores for output channels
        f_scores = fisher.get(name, torch.ones(out_c))
        if isinstance(f_scores, torch.Tensor):
            f_scores = f_scores.cpu().numpy()

        # Sort channels by Fisher importance
        sorted_channels = sorted(enumerate(f_scores), key=lambda x: x[1], reverse=True)

        # Stage 2: Allocate precision
        allocation = allocate_precision(sorted_channels, int4_fraction)

        # Stage 3: Select blocksize for binary channels
        binary_channels = [i for i, a in allocation.items() if a == 'binary']
        int4_channels = [i for i, a in allocation.items() if a == 'int4']

        if binary_channels:
            binary_weights = weights[binary_channels, :]  # (nb, in_c)
            binary_fisher = f_scores[binary_channels]
            best_bs = select_best_blocksize(binary_weights, binary_fisher, blocksize_candidates)
        else:
            best_bs = 128  # fallback

        # Stage 4: Build residual codebook and quantize
        # ... (full implementation in notebook)

        quantized_layers[name] = QuantizedLayer(...)

    return quantized_layers
```

---

## Evaluation Strategy

### Perplexity (Primary Metric)

Evaluate on held-out text (WikiText-2, C4, or Pile subset).

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

### Baselines to Compare

| Method | Description |
|--------|------------|
| **FP16** | Full precision baseline |
| **Q1_0_g128** | llama.cpp Q1_0_g128 format (Bonsai's approach) |
| **BiLLM** | Hessian-based 1-bit PTQ (if open-source code available) |
| **FABQ-RC (ours)** | Fisher-adaptive with residual codebook |

---

## Expected Results

Based on corrected analysis:

| Method | bpw | Perplexity (7B) | Est. Quality Gap |
|--------|-----|-----------------|------------------|
| FP16 | 16.0 | baseline | — |
| Q1_0_g128 | 1.125 | poor | large |
| BiLLM | 1.08 | 8.41 (70B) | small |
| **FABQ-RC (target)** | ~1.21 | < 8.0 | smallest |

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

## Out of Scope for v1

- Activation quantization (weight-only for now)
- Hardware-aware blocksize (GPU memory coalescing)
- Per-token vs per-block scale optimization
- Fine-tuning after quantization (QAT)

---

## Changelog (v1 → v2)

1. **int8 → int4 for top 5% channels**: saves 0.20 bits/weight
2. **Dropped blocksize 16, 32**: minimum blocksize 64 to avoid scale overhead inflation
3. **Single global codebook → 4 tiered codebooks (64 centroids each)**: better residual coverage per Fisher tier
4. **uint8 indices → 4-bit indices**: 16 centroids per layer cluster, halves codebook overhead
5. **Corrected bpw math**: Total now ~1.21 bpw (was incorrectly claimed as ~1.38 with optimistic int8 accounting)
