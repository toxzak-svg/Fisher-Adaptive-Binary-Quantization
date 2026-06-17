---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
tags:
- quantization
- 1-bit
- gguf
- llm
- fabq-rc
- fisher-adaptive
library_name: llama.cpp
---

# Qwen3.6-27B-FABQ-RC-GGUF

## Model Description

FABQ-RC (Fisher-Adaptive Binary Quantization with Residual Codebooks) is a 1-bit quantization method for large language models that adapts per layer rather than using a fixed blocksize.

FABQ-RC combines four innovations:
1. **Fisher-Weighted Channel Importance** — Uses Fisher Information to determine which channels actually matter for loss
2. **Mixed-Precision Core Allocation** — int4 for critical channels, binary ±1 for the rest
3. **Adaptive Blocksize** — Per-layer blocksize selection {64, 128, 256, 512}
4. **Residual Codebook** — k-means clustering corrects systematic binary quantization bias

## Quantization Details

| Property | Value |
|---------|-------|
| **Base Model** | Qwen/Qwen3.6-27B |
| **Format** | GGUF |
| **Bits per parameter** | ~1.18 bpw |
| **Quantization Method** | FABQ-RC |
| **Calibration Dataset** | C4 (2048 samples, seq_len=32) |
| **Precision Allocation** | Top 5% channels → int4, 95% → binary |
| **Blocksize Selection** | Adaptive per-layer |

## How It Works

### Stage 1: Fisher-Weighted Channel Importance
Fisher Information measures expected gradient² per channel, telling us which weights actually matter for the loss. This is more directly relevant than Hessian (curvature) or magnitude.

### Stage 2: Mixed-Precision Allocation
Top 5% Fisher channels → int4 (preserve accuracy). Bottom 95% → binary ±1 (maximum compression).

### Stage 3: Adaptive Blocksize
Each layer gets its own optimal blocksize from {64, 128, 256, 512}, chosen by minimizing Fisher-weighted reconstruction error. Some layers need fine-grained blocks; others can use larger blocks.

### Stage 4: Residual Codebook
After binary quantization, systematic residuals remain. FABQ-RC clusters these residuals using 4 tiered k-means codebooks (64 centroids each), enabling non-linear correction that beats BiLLM's linear approximation.

## Use with llama.cpp

### CLI Inference
```bash
# Download and run
./llama-cli -m Qwen3.6-27B-FABQ-RC-Q4_K_M.gguf -n 256 -p "The future of 1-bit quantization is"

# Interactive mode
./llama-cli -m Qwen3.6-27B-FABQ-RC-Q4_K_M.gguf -i -Ins 256
```

### Perplexity Evaluation
```bash
./llama-perplexity -m Qwen3.6-27B-FABQ-RC-Q4_K_M.gguf -f wikitext.txt
```

### Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(
    model_path="Qwen3.6-27B-FABQ-RC-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # Auto-detect
)

output = llm(
    "The future of 1-bit quantization is",
    max_tokens=256,
    temperature=0.7,
)
print(output['choices'][0]['text'])
```

## Benchmark Results

| Method | bpw | Perplexity | Notes |
|--------|-----|------------|-------|
| FP16 | 16.0 | baseline | Qwen3.6-27B |
| **FABQ-RC** | ~1.18 | TBD | Our method |

## Limitations

- **Weight-only quantization**: Activations are not quantized
- **Short calibration sequences**: 32 token context length may miss long-range dependencies
- **Single-domain calibration**: C4 only; may not generalize perfectly to other domains

## Training Details

- **Method**: FABQ-RC (see [FABQ-RC specification](https://github.com/toxzak/fabq-rc))
- **Calibration**: C4 dataset, 2048 samples
- **Hardware**: A100 80GB GPU

## GitHub Repository

Full implementation, training code, and documentation available at:
[https://github.com/toxzak/fabq-rc](https://github.com/toxzak/fabq-rc)

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

- GitHub Issues: [https://github.com/toxzak/fabq-rc/issues](https://github.com/toxzak/fabq-rc/issues)
- HuggingFace: [toxzak](https://huggingface.co/toxzak)

---

*Built with ❤️ by Zach Maronek · 2026*
