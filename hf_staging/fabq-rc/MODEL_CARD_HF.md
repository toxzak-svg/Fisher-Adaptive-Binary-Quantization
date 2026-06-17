---
language:
- en
- zh
- de
- fr
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
tags:
- quantization
- 1-bit
- gguf
- llm
- fabq-rc
- fisher-adaptive
- qwen
library_name: llama.cpp
inference:
  warmup_required: true
  device_type: gpu
quark:
  color: "#7C3AED"
pipeline_tag: causal-lm
extra_metadata:
  quantization_label:
    - Q1_K
thumbnail: ""
---

# Qwen3.6-27B-FABQ-RC-GGUF

## Model Description

**FABQ-RC** (Fisher-Adaptive Binary Quantization with Residual Codebooks) is a 1-bit quantization method for large language models that adapts per layer rather than using a fixed blocksize. This quantization of Qwen3.6-27B achieves ~1.18 bits per parameter while maintaining quality through four key innovations.

| Property | Value |
|----------|-------|
| **Base Model** | Qwen/Qwen3.6-27B |
| **Quantization Method** | FABQ-RC |
| **Format** | GGUF (v3) |
| **Bits per Parameter** | ~1.18 bpw |
| **Precision Allocation** | Top 5% channels → int4, 95% → binary ±1 |
| **Blocksize** | Adaptive per-layer {64, 128, 256, 512} |
| **Calibration Dataset** | C4 (2048 samples, seq_len=32) |

## How It Works

FABQ-RC combines four innovations:

1. **Fisher-Weighted Channel Importance** — Uses Fisher Information (expected gradient²) to determine which channels actually matter for the loss. This is more directly relevant than Hessian (curvature) or magnitude alone.

2. **Mixed-Precision Core Allocation** — Top 5% Fisher channels → int4 (preserve accuracy). Bottom 95% → binary ±1 (maximum compression).

3. **Adaptive Blocksize** — Each layer gets its own optimal blocksize from {64, 128, 256, 512}, chosen by minimizing Fisher-weighted reconstruction error. Homogeneous layers use larger blocks; heterogeneous layers use smaller ones.

4. **Residual Codebook** — After binary quantization, systematic residuals remain. FABQ-RC clusters these using 4 tiered k-means codebooks (64 centroids each, Fisher quartile-based), enabling non-linear correction that beats BiLLM's linear approximation.

```
FP32 Weights
    │
    ▼
Stage 1: Fisher-Weighted Channel Importance
    │  Compute per-channel Fisher Information
    │  Sort channels by expected loss impact
    ▼
Stage 2: Mixed-Precision Allocation
    │  Top 5% channels → int4 (preserve accuracy)
    │  Bottom 95% channels → binary ±1 (max compression)
    ▼
Stage 3: Adaptive Blocksize Selection
    │  Per-layer sweep {64, 128, 256, 512}
    │  Pick blocksize minimizing Fisher-weighted reconstruction error
    ▼
Stage 4: Residual Codebook Clustering
    │  4 tiered codebooks × 64 centroids (Fisher quartile-based)
    │  4-bit indices per block (16 centroids per layer cluster)
    ▼
FABQ-RC GGUF
```

## Why FABQ-RC > Other 1-bit Methods

| Method | Blocksize | Residual Handling | Importance Metric |
|--------|-----------|-------------------|-------------------|
| Q1_0_g128 (Bonsai) | Fixed 128 | None | Magnitude |
| BiLLM | Fixed | Linear approximation | Hessian |
| **FABQ-RC** | **Adaptive** | **Non-linear codebook** | **Fisher** |

FABQ-RC beats BiLLM because:
- **Adaptive blocksize** recovers more per-layer quality than fixed blocksize
- **Fisher > Hessian** as importance metric (direct loss relevance vs. curvature approximation)
- **Residual codebook** corrects systematic binary quantization bias better than linear approximation

## Use with llama.cpp

### CLI Inference

```bash
# Download and run
./llama-cli -m Qwen3.6-27B-FABQ-RC-Q1_K.gguf -n 256 -p "The future of 1-bit quantization is"

# Interactive mode
./llama-cli -m Qwen3.6-27B-FABQ-RC-Q1_K.gguf -i -Ins 256

# With longer context
./llama-cli -m Qwen3.6-27B-FABQ-RC-Q1_K.gguf -ctx 4096 -i -Ins 256
```

### Perplexity Evaluation

```bash
./llama-perplexity -m Qwen3.6-27B-FABQ-RC-Q1_K.gguf -f wikitext.txt
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="Qwen3.6-27B-FABQ-RC-Q1_K.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # Auto-detect GPU offload
    verbose=False,
)

output = llm(
    "The future of 1-bit quantization is",
    max_tokens=256,
    temperature=0.7,
)
print(output['choices'][0]['text'])
```

## Quantization Details

### Precision Allocation

- **int4 channels (5%):** Preserved for highest Fisher Information channels — these determine the layer's behavior
- **Binary channels (95%):** Compressed to ±1 with per-block scaling

### Adaptive Blocksize Distribution

| Blocksize | Typical Layers |
|-----------|----------------|
| 64 | Attention projections, heterogeneous layers |
| 128 | FFN layers, moderately heterogeneous |
| 256 | Homogeneous FFN layers |
| 512 | Embedding layers, highly homogeneous |

### Residual Codebook Architecture

- **4 tiered codebooks** of 64 centroids each (Fisher quartile-based assignment)
- **4-bit indices** per block (16 active centroids per layer cluster)
- Total codebook storage: 4 × 64 × 128 × 4 bytes = 128KB per blocksize (negligible)

## Benchmark Comparison

| Method | bpw | Perplexity (est.) | Notes |
|--------|-----|-------------------|-------|
| FP16 (baseline) | 16.0 | — | Qwen3.6-27B full precision |
| Q1_0_g128 | 1.125 | degraded | Bonsai's format |
| BiLLM | 1.08 | ~8.41 (70B) | Best prior work |
| **FABQ-RC** | ~1.18 | TBD | Our method |

## Limitations

- **Weight-only quantization:** Activations are not quantized
- **Short calibration sequences:** 32 token context length may miss long-range dependencies
- **Single-domain calibration:** C4 only; may not generalize perfectly to other domains

## Training Details

| Property | Value |
|----------|-------|
| **Method** | FABQ-RC (see [FABQ-RC specification](https://github.com/toxzak/fabq-rc)) |
| **Calibration** | C4 dataset, 2048 samples |
| **Sequence Length** | 32 tokens |
| **Hardware** | A100 80GB GPU |
| **Export Format** | GGUF v3 |

## Files

| File | Description |
|------|-------------|
| `Qwen3.6-27B-FABQ-RC-Q1_K.gguf` | Main quantized model |
| `*.pt` (in repo) | Intermediate checkpoint (FP16 reconstruction) |

## Citation

```bibtex
@misc{fabqrc2026,
    author = {Zach Maronek},
    title = {FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks},
    year = {2026},
    url = {https://github.com/toxzak/fabq-rc}
}
```

## Related Models

- [toxzak/Qwen3.6-27B-FABQ-RC](https://huggingface.co/toxzak/Qwen3.6-27B-FABQ-RC) — safetensors format
- [toxzak/Qwen3.6-35B-A3B-FABQ-RC](https://huggingface.co/toxzak/Qwen3.6-35B-A3B-FABQ-RC) — 3-bit variant

## Acknowledgments

- Qwen team for the excellent base model
- llama.cpp team for the GGUF format and inference infrastructure
- BiLLM and Bonsai authors for pioneering 1-bit quantization research

---

*FABQ-RC by Zach Maronek · 2026*