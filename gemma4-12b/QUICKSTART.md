# Quick start — FABQ-RC Gemma 4 12B

Four commands. That's the whole project.

## 0. One-time setup

```bash
cd gemma4-12b
cp .env.example .env
# Edit .env and paste your HF_TOKEN
```

Get a token at https://huggingface.co/settings/tokens. The
`google/gemma-4-12B-it` model is gated — you also need to accept the
license on its model page before the token will work.

## 1. Install

```bash
make install
```

Installs everything in `requirements.txt`. ~2 min.

## 2. Pick your path

### Path A: just want to see FABQ-RC work (30-60 min on A100)

```bash
make notebook-text
```

This opens `FABQ-RC-Gemma4-12B.ipynb`. It's the simple text quantization
notebook — loads the model, runs the 4-stage pipeline, reports bpw and
perplexity. No CUDA extension build, no bucket needed.

### Path B: full native-quantized inference demo

Three steps:

```bash
# 2a. Verify the CUDA extension builds (2-5 min)
make test-cuda

# 2b. Populate the HF bucket (one-time, ~30-45 min on A100)
make bucket

# 2c. Launch the streaming notebook
make notebook-stream
```

The streaming notebook streams the pre-quantized shards from the bucket
and runs inference on the CUDA kernel. Peak VRAM is ~2.2 GB (vs 24 GB
for BF16) because the FP16 weight is never materialized.

## 3. Re-push (if you edited anything)

```bash
make push                # actual upload
make push-dry-run        # see what would be uploaded first
```

## 4. Clean up

```bash
make clean               # trash staging dir + extension build artifacts
```

---

## What you need

**Hardware:**
- A100 80GB (or any GPU with >=24 GB VRAM for the text notebook; >=8 GB
  for the streaming variant after the model is loaded)
- ~50 GB free disk for the BF16 source during bucket build
- CUDA toolkit >= 12.1 (only for `make test-cuda` and `make bucket`)

**Access:**
- HuggingFace account with accepted Gemma license
- HF_TOKEN with read scope (gated model)

**What you DON'T need:**
- No Llama.cpp, no bitsandbytes at runtime (the CUDA extension does the
  matmul itself)
- No custom training data
- No GPU-specific knowledge beyond the basics

## Common questions

**Q: The notebook hangs on the first import.**
A: Restart and run cell-by-cell. Colab sometimes hangs on
`!pip install` if the runtime is cold.

**Q: `make test-cuda` fails with "nvcc not found".**
A: You need the CUDA toolkit in PATH. Set `CUDA_HOME` and add
`$CUDA_HOME/bin` to PATH. On Colab this Just Works; on local you need
to install CUDA 12.1+ first.

**Q: The kernel numerical test fails with a small max-diff.**
A: Expected — fp16 accumulation has some loss. The threshold is loose
on purpose. If it fails with `max diff > 0.5`, that's a real bug, ping
the maintainer.

**Q: Do I need both notebooks?**
A: No. The text notebook is the research one. The streaming notebook
is the production one. They share the same on-disk format and the same
quantization pipeline, but the streaming one needs the bucket.

**Q: How long does `make bucket` take?**
A: ~30-45 min on A100. ~24 GB BF16 download + calibration + quantize
+ upload. The 2048 C4 calibration samples are the calibration; the
k-means codebook is the slow part of quant.

**Q: Can I re-quantize with different calibration data?**
A: Yes. `make bucket CALIB_SAMPLES=4096 MAX_SEQ_LEN=1024` (or edit
`build_bucket.py` defaults). The bucket is a *rebuild*, not an
append — you get a fresh stats file and fresh shards.

## Files in this folder

```
gemma4-12b/
├── Makefile                            # you are here
├── QUICKSTART.md                       # this file
├── requirements.txt                    # pip install -r requirements.txt
├── .env.example                        # copy to .env, set HF_TOKEN
├── README.md                           # full README
├── MODEL_CARD.md                       # HF model card
├── FABQ_RC_GGUF_SPEC.md                # GGUF format spec
│
├── FABQ-RC-Gemma4-12B.ipynb            # text-only quantization notebook
├── build_notebook.py                   # regenerates the .ipynb
├── push_to_hf.py                       # curate + push to HF
│
└── streaming/                          # native-quantized inference
    ├── fabq_rc_cuda/                   # C++/CUDA extension
    ├── build_bucket.py                 # populates the HF bucket
    ├── build_streaming_notebook.py
    ├── FABQ-RC-Gemma4-12B-Streaming.ipynb
    └── STREAMING.md                    # design doc
```

## License

Apache 2.0
