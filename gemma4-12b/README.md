---
license: apache-2.0
base_model: google/gemma-4-12B-it
tags:
  - quantization
  - 1-bit
  - fabq-rc
  - fisher-adaptive
  - gemma
  - native-quantized-inference
  - cuda-kernel
  - code
pipeline_tag: other
library_name: fabq-rc
---

# FABQ-RC — Gemma 4 12B

This folder contains the FABQ-RC quantization pipeline for **Gemma 4 12B-it**.

It includes two variants:

1. **Text quantization notebook** (`FABQ-RC-Gemma4-12B.ipynb`) — quantizes
   the model, reports bpw, runs perplexity. Uses the standard
   "reconstruct FP16 then matmul" path. Good for ablation studies.

2. **Streaming + native-quantized inference** (`streaming/`) — runs the
   model where the forward pass operates directly on the compressed
   FABQ-RC buffers, never materializing the FP16 weight matrix. Peak
   inference VRAM is ~2.2 GB (vs 24 GB for BF16).

The base notebook was adapted from:

- `../notebooks/latest/Main_FABQ_RC_Notebook.ipynb` — the Qwen3.6-27B baseline
- `notebooks/build_v4_flash_notebook.py` — the cleaner script-style
  structure used to build the DeepSeek V4-Flash notebook

## Quick start

### Option A: text-only quantization (simple)

Open `FABQ-RC-Gemma4-12B.ipynb` in Colab (A100 80GB) and run all cells.
The notebook is self-contained.

### Option B: streaming + native-quantized inference

This is the variant where the forward pass reads the compressed FABQ-RC
buffers directly. It needs a working CUDA toolchain.

```bash
# 1. Build the bucket (one-time, ~30-45 min on A100)
cd streaming
python build_bucket.py --source google/gemma-4-12B-it --push

# 2. Run the streaming notebook
jupyter notebook FABQ-RC-Gemma4-12B-Streaming.ipynb
```

The notebook is self-contained: it builds the CUDA extension in-cell
on first run, then streams the pre-quantized shards from
`toxzak/gemma-4-12B-it-fabq-rc-bucket`.

## Gemma 4 12B specifics

- **48 decoder layers**, hidden 3840, intermediate 15360
- **Tied embeddings** (`tie_word_embeddings: true`): 262144 × 3840 =
  ~1B params, ~2 GB in BF16. We skip the embedding in the linear sweep
  and keep it in BF16 for the output logits.
- **Hybrid attention**: 6 full-attention layers (every 6th) + the rest
  sliding-window 1024. Proportional RoPE on full, regular RoPE on sliding.
- **Multimodal** (vision + audio) — out of scope for v1 of the streaming
  variant. Text-only path; the multimodal encoders stay in BF16 if loaded.

## File layout

```
gemma4-12b/
├── FABQ-RC-Gemma4-12B.ipynb          # text-only quantization (variant 1)
├── build_notebook.py
├── README.md
├── MODEL_CARD.md
├── FABQ_RC_GGUF_SPEC.md
└── streaming/                         # native-quantized inference (variant 2)
    ├── fabq_rc_cuda/                  # C++/CUDA extension
    ├── build_bucket.py                # one-time bucket build
    ├── build_streaming_notebook.py
    ├── FABQ-RC-Gemma4-12B-Streaming.ipynb
    └── STREAMING.md                   # design doc
```

## Why two variants

The text notebook is the "research" path: standard FP16 reconstruction,
easy to ablate, easy to compare to other quant methods.

The streaming variant is the "production" path: peak inference memory
is the compressed size (~2.2 GB vs 24 GB), no FP16 weight ever lives
in VRAM during inference, ready to plug into an edge deployment or
a small-GPU serving setup.

Both share the same underlying FABQ-RC quantization pipeline and the
same on-disk format.

## License

Apache 2.0
