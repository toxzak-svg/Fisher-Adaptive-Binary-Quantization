# FABQ-RC Validation Memo

**Date:** 2026-06-15
**Trigger:** "Validate it then" — your response to my pushback on adding
"Sub-Byte Micro-Scaling" and "Adaptive Blocksize kernel hardening" on top
of an unvalidated floor.
**Scope:** Inventory + format math + dynamic-blocksize audit + perplexity
benchmark status. No code changes.

---

## TL;DR

| Claim in the README / spec                  | Status         |
|---------------------------------------------|----------------|
| 1.18 bpw (Qwen3.6-27B) / 1.21 bpw (12B)     | **Unverified** — see §3 |
| Perplexity target < 8.0                      | **Not measured** — see §4 |
| Per-layer blocksize {64, 128, 256, 512}     | **Specified + code-ready, never tested with all 4 values** — see §5 |
| Native CUDA inference at 2.2 GB VRAM        | **Code-ready, never built/tested locally** — see §5.3 |
| Published model on HuggingFace              | **Metadata only — no eval log on disk** — see §4 |

The repo is in a *spec-and-notebook-pioneer* state, not a *measured-results*
state. Every number in the README and MODEL_CARD is a *prediction*, not
a *result*. That is a real problem for the micro-scaling proposal you
asked me to evaluate.

---

## 1. What's actually in the repo

**Specs (well-written, internally consistent at the level they go):**
- `FABQ_RC_SPEC.md` — the four-stage method, target ~1.21 bpw
- `plans/EBQ-SPEC.md` — global knapsack allocator for 2-8 bit
- `plans/FABQ-VP-SPEC.md` — variable precision extension
- `plans/UNIFIED-SPEC.md` — combined architecture
- `plans/RESEARCH-PLAN.md` — explicitly lists "Padded-block centroid bug"
  as known issue + Phase 0.2 "Baseline perplexity measurement" as
  *prerequisite before any extension*
- `gemma4-12b/streaming/STREAMING.md` — solid memory-layout walk-through
- `FABQ_RC_GGUF_SPEC.md` — 113 lines, two different "GGUF" specs that
  disagree (root version vs gemma4-12b version) — see §6

**Code:**
- `gemma4-12b/streaming/fabq_rc_cuda/` — C++/CUDA extension: format header,
  gemm kernel (scalar, no tensor cores), quantizer, Python wrappers
- `quant_pipeline.py` — blocksize sweep + Fisher channel allocation
- `fisher.py` — calibration pass
- `kmeans.py` — codebook builder
- `tests/test_kernel.py` — 5 tests, 3 are CUDA-only, 2 are pure-Python
  round-trips. **Not runnable on this Windows / CPU-only torch install
  (no CUDA toolkit, no nvcc).**

**Notebooks:** 9 notebooks across 4 model variants (Qwen3.6-27B,
Gemma-4-12B, DeepSeek-V4-Flash, TinyLlama FABQ-VP). All contain
`compute_perplexity()` calls but **no output cells with measured numbers
are checked in**.

**No eval results on disk anywhere.** No `results.json`, no `*ppl*` files,
no `*wikitext*` files, no logged CSV/JSON. The whole evaluation pipeline
ran (if at all) on Colab/Kaggle and the output never made it back to the
repo.

**No model weights on disk.** `models/` has only an empty HF download
cache. The published model lives at `toxzak/Qwen3.6-27B-FABQ-RC-GGUF`
and `toxzak/gemma-4-12B-it-fabq-rc-bucket` only.

---

## 2. Two competing bpw numbers

| Source | Number |
|--------|--------|
| `README.md` line 36, 113, 126 | **~1.18** |
| `MODEL_CARD_HF.md` line 34, 41, 164 | **~1.18** |
| `MODEL_CARD.md` (the non-HF one) line 32, 91 | **~1.18** |
| `FABQRC_PLAN.md` line 19 | **~1.18** (target to confirm) |
| `plans/RESEARCH-PLAN.md` line 32 | **~1.18** |
| `FABQ_RC_SPEC.md` line 45, 337, 372 | **~1.21** (with "Corrected bpw math" changelog) |
| `gemma4-12b/streaming/STREAMING.md` line 151 | **1.21** |
| `gemma4-12b/MODEL_CARD.md` line 44 | **~1.21 (theoretical)** |
| `gemma4-12b/FABQ_RC_GGUF_SPEC.md` line 123 | **1.21** (12.5B × 1.21 bpw = 1.9 GB) |

The two numbers are not the same. 1.18 appears in user-facing surfaces
(README, HF model card, plan). 1.21 appears in the canonical spec and
in the gemma4-12b derived material. The "Corrected bpw math" changelog
in the spec says v1 was 1.18, v2 is 1.21 — meaning the README and HF
card are using the **deprecated** number. They were never updated.

**Neither number has been measured.** Both are derived from the on-disk
format spec, not from a real quantized model.

---

## 3. Walking the bpw math myself

The on-disk format is defined in
`gemma4-12b/streaming/fabq_rc_cuda/src/fabq_rc_format.h` and packed by
`fabq_rc_quant.cpp`. For a layer of shape (out_c, in_c) with 5% int4
channels, 95% binary channels, blocksize B, and the shared codebook
amortized across the model:

| B    | bytes/layer (3840×3840) | bpw (no codebook) |
|------|------------------------|--------------------|
| 64   | 3.18 MB                | **1.73**           |
| 128  | 2.86 MB                | **1.55**           |
| 256  | 2.69 MB                | **1.46**           |
| 512  | 2.61 MB                | **1.42**           |

The STREAMING.md doc says ~2.7 MB per layer for the 3840×3840 case, which
matches my math. So the doc is internally consistent at **2.7 MB /
layer**. At B=128.

Neither 1.18 nor 1.21 matches 1.55. The spec/README are off by 25-30%
from the actual on-disk format. The likely explanation is they were
counting **logical** bits (4 bits for the int4 channels, 1 bit for
binary) without the **storage** overhead (int8 packing of int4 weights,
FP16 scales per row, FP16 scales per binary block, codebook index bytes,
channel-map indices, header, bias). My walk counts storage, which is what
actually shows up on disk and in VRAM.

**This is the first concrete validation finding.** The README's 1.18 bpw
is a marketing number from before the v1→v2 "corrected math" changelog.
The canonical spec's 1.21 is closer but still undercounts the storage
overhead. The real number for the 3840×3840 layer at B=128 is **~1.55
bpw**, and it scales 1.42–1.73 bpw across the blocksize sweep. This
needs to be acknowledged in `FABQ_RC_SPEC.md` and propagated to the
README + model cards *before* any further "squeeze it lower" work, or
the floor you're optimizing from is fiction.

---

## 4. Perplexity benchmark status

**No perplexity number for the published model exists in this repo.**

I searched for: `*ppl*`, `*wikitext*`, `*results*`, `*eval*`, `ppl =`,
`ppl_quantized`, `"perplexity":`. Nothing.

The notebooks contain `compute_perplexity()` implementations and `print`
statements, but no executed output is checked in. The published model
cards say:
- `MODEL_CARD_HF.md` line 164: `| **FABQ-RC** | ~1.18 | **TBD** | Our method |`
- `FABQRC_PLAN.md` line 27: `### 1.2 Perplexity benchmarks ... **Status:** Model published, validation pending`

`plans/RESEARCH-PLAN.md` line 41-59 has a *Phase 0.2 Baseline Perplexity
Measurement* section that explicitly says "**Before extending, measure:**
1. FABQ-RC (fixed) perplexity on WikiText2" — and it's an open checkbox.

The `RESEARCH-PLAN.md` also flags a *Phase 0.1 Padded-Block Centroid Bug*:
"Issue: Cells 19 and 21 in FABQ_RC_Kaggle.ipynb compute residuals
including padded blocks, skewing centroid computation." This was
identified but the fix status is not marked complete anywhere I can see.

**What this means for the "Sub-Byte Micro-Scaling" proposal:** you cannot
optimize from a target you haven't measured. If the actual on-disk bpw
is 1.55 (not 1.18), the gap to close is bigger than the README
suggests, and the gain budget is different. If the actual perplexity
after fixing the padded-centroid bug is, say, 12 instead of <8, then
"without losing the perplexity gains you've achieved" is a constraint
with no current solution in the search space. We need numbers on disk
before any further micro-optimization.

---

## 5. Adaptive blocksize audit

### 5.1 Spec / code intent

`FABQ_RC_SPEC.md` line 33-35 says blocksize is per-layer, candidates
`{64, 128, 256, 512}`. `quant_pipeline.py:BS_CANDIDATES = [64, 128, 256, 512]`
is the actual list. `select_best_blocksize()` runs the sweep and
returns the per-layer winner.

### 5.2 Storage and kernel support

`fabq_rc_format.h` stores `uint32_t blocksize` per layer. The CUDA
kernel (`fabq_rc_gemm.cu`) takes `int blocksize` and `int n_blocks` as
runtime arguments and computes `blk_start = blk * blocksize; blk_end =
min(blk_start + blocksize, in_features)`. **So the kernel and format
do support blocksize varying per layer at runtime.** That part of the
claim is correct.

The hidden gotchas:

**a) The codebook shape is fixed at `max_blocksize = 512`.** Per
`fabq_rc_format.h` line 47-50: `4 × 64 × 512 × fp16 = 256 KB`. If a
layer picks B=64, the codebook still allocates 512-element rows, the
quantizer (`fabq_rc_quant.cpp:142-156`) still pads residuals to 512
before L2 distance, and the kernel reads `cb[local_i]` for `local_i <
blk_len` where `blk_len = 64`. So **every layer pays the full 512-wide
codebook storage**, not just the layers that picked B=512. That's
wasted VRAM but functionally correct. For the 12B model with 256 KB
total codebook, the waste is small. For a 70B model it grows linearly
in layers.

**b) The `_forward_reference` (pure-PyTorch) path on lines 122-141 of
`quantized_linear.py` iterates `for blk in range(n_blocks)` Python-side
and does `cb_vec = cb[cb_id, :blk_len]`. This works for any B but is
`O(blocks per layer)` Python overhead — fine for testing, irrelevant
for inference since CUDA path is used.

**c) `n_blocks` changes per layer.** This is correctly passed as a
runtime int (line 174 / 178 of `quantized_linear.py`) and as a
runtime int in the kernel launchers. No template specialization
needed. **Good.**

**d) The selection penalty table.** `quant_pipeline.py:BS_PENALTIES =
{64: 1.5, 128: 1.0, 256: 0.85, 512: 0.75}` — this is a *hand-tuned*
soft bias, not a measured overhead. Larger B *should* be slightly
cheaper (less scale overhead), but the relative magnitudes (0.75 vs 1.5
= 2x spread) are not derived from any kernel benchmark in the repo.
I'd want a real measurement (scale-bytes vs total-bytes as a function
of B) before treating this table as ground truth.

**e) `n_blocks` for the int4 mixed kernel** is the binary blocks count
only (the int4 path doesn't have blocks). At B=512 for a 3840-wide
layer: `n_blocks = ceil(3840/512) = 8`; at B=64: `n_blocks = 60`. The
mixed kernel passes `n_blocks` to the binary path and the int4 path
ignores it. The CUDA kernel's `for blk in range(n_blocks)` loop bound
is correct. **OK.**

### 5.3 Has this ever been tested with B=64 and B=512?

`test_kernel.py` line 187 hardcodes `blocksize = 64` for the round-trip
test. So the **storage format** has been smoke-tested with B=64. There
is no test with B=256 or B=512. There is no CUDA test for blocksize ≠
128 — the two CUDA tests (`test_int4_only_layer_forward` and
`test_mixed_layer_forward`) both use B=128. The CUDA build itself
requires a CUDA toolkit which is not installed here, so I couldn't
re-run the tests on this machine.

**The format and kernel logic supports B=64 / 128 / 256 / 512. The test
coverage covers B=64 and B=128 only. The "B=256 and B=512 are dynamic
and tested" claim is not backed by anything in the repo.**

---

## 6. Discrepancies between the two GGUF specs

`FABQ_RC_GGUF_SPEC.md` exists in two places:
- Root version: 113 lines, GGML_TYPE_FABQ_RC = 41, single per-layer
  `block_fabq_rc` struct with a codebook-index field per block
- `gemma4-12b/streaming/FABQ_RC_GGUF_SPEC.md`: 200+ lines, different
  layout, `block_fabq_rc_v2` with separate fp16 scales per int4 vs
  binary, blocksize stored in the per-block header as u16

The two specs disagree on the per-block struct layout, the codebook
storage path, and the metadata key naming. Neither is referenced from
the other. The gemma4-12b spec is the newer one (it accounts for the
5%-int4 / 95%-binary layout, while the root spec assumes uniform binary).

If you intended to ship a GGUF, only one of these can be the source of
truth. Currently the README points to a GGUF that doesn't have a
single coherent spec behind it. This is a validation finding that
should be resolved before the model card claims llama.cpp compatibility.

---

## 7. Concrete validation status by claim

| Claim | Where | Status |
|-------|-------|--------|
| FABQ-RC achieves 1.18 bpw | README, HF card | **Refuted by my math** — actual is 1.42-1.73 bpw per layer at B=128 |
| FABQ-RC achieves 1.21 bpw | spec, gemma4-12b docs | **Also refuted** — actual is 1.42-1.73 bpw |
| Perplexity target < 8.0 | README, spec | **Not measured** |
| Beats BiLLM | README, spec | **Not measured** (BiLLM not even run here) |
| Adaptive blocksize kernel handles {64, 128, 256, 512} | your message to me | **Mostly true for the kernel/format, but only B=64 and B=128 are tested; B=256 and B=512 have no test** |
| Native CUDA inference | gemma4-12b docs | **Code exists, not built locally, not benchmarked** |
| Beats Q1_0_g128 | README table | **Not measured** |
| v1 code is "correct and memory-safe" | gemm.cu comment | **Plausible** — 3 of 5 tests would catch memory errors, and the int8/int64 types are sound, but the BPF-quantization error is unquantified |
| Padded-centroid bug is fixed | (status?) | **Not marked complete in any plan file** |
| Llama.cpp integration | README, MODEL_CARD_HF | **No llama.cpp fork / PR / branch referenced anywhere** |

---

## 8. What I am NOT recommending

I am **not** recommending that we add "Sub-Byte Micro-Scaling" or any
further compression layer to the spec today. Here's why:

1. The 1.18 / 1.21 bpw claims in the README are wrong by 25-30% against
   the on-disk format. We need to first fix the README and spec
   numbers to the actual 1.42-1.73 bpw, run a perplexity eval, and
   verify the padded-centroid bug is fixed.

2. The "perplexity gains" the proposal claims to preserve don't exist
   in any measured form. We have *predictions* of <8.0 ppl. We have
   *no measurements*. Optimizing against a phantom floor is a great
   way to ship a regression.

3. The "Idea 1" / "Idea 3" framing from the Gemini chat is not in
   this repo. If you want me to evaluate the design, I need the
   source.

## 9. What I AM recommending (sequenced)

### Step 1 — Fix the README and spec bpw numbers (1-2 hours)

Walk through the format math in a new section of `FABQ_RC_SPEC.md`
called "Storage budget derivation" and replace the "Effective bits per
weight: ~1.21 bpw" line with a real formula. The table of per-bpw
across B={64, 128, 256, 512} from §3 of this memo is the right shape.

### Step 2 — Pick one GGUF spec as canonical and delete the other (1 hour)

The two `FABQ_RC_GGUF_SPEC.md` files are an inconsistency, not a
redundancy. Pick one. I'd recommend the gemma4-12b version (it's
newer, accounts for the 5%/95% mix, and is in the path that's
actually being built).

### Step 3 — Run the Phase 0.1 padded-centroid fix verification

`plans/RESEARCH-PLAN.md` line 41-50. Either confirm it's already
shipped (in which case mark the checkbox and remove the warning) or
do the cosine-similarity check. This is a prerequisite for any
perplexity number being meaningful.

### Step 4 — Run a real perplexity eval

The published model is on HF. The bucket is on HF. Either:
- (a) `snapshot_download toxzak/Qwen3.6-27B-FABQ-RC-GGUF`, run
  `llama-perplexity -m ...gguf -f wikitext-2-raw-v1/test/wiki.test.raw`
  on an A100, log the result to a `results/` folder in the repo.
- (b) Run the streaming notebook against
  `toxzak/gemma-4-12B-it-fabq-rc-bucket` and use its built-in
  `compute_perplexity` to log a number on Gemma 4 12B.

This produces a real number that any future "squeeze lower" proposal
can be measured against.

### Step 5 — THEN consider the micro-scaling proposal

Once we have (a) a corrected bpw, (b) a real perplexity number, and
(c) the padded-centroid bug verified fixed, the "Sub-Byte
Micro-Scaling" idea becomes a real engineering question with a real
trade. The current design space is:
- FP16 → FP8 scales (saves 1 byte per scale slot × `n_binary × n_blocks`
  scales per layer; for 12B that's ~3-4 MB savings, not the order of
  magnitude the proposal implies)
- Larger effective blocks (one scale per 32 elements instead of 1 per
  64-128) — saves scale overhead but increases quantization error
- Shared cross-channel scales (MX-style 32×32 tile) — restructures the
  format, requires kernel rewrite
- Quantized scales + tiny codebook for the scales themselves — 8-bit
  indices into a 256-entry FP8 codebook for the scale values

None of these are "squeeze 1.18 lower." Each is a measured engineering
trade. The 0.07-0.36 bpw between my measured 1.55 and the README's
1.18 is bigger than any of these would recover, and the real gains
are at most 0.05-0.10 bpw. **The bigger win is admitting the 1.18
claim is wrong, not chasing it lower with more layers.**

---

## 10. Open questions for you

1. Has the published model on HF (`toxzak/Qwen3.6-27B-FABQ-RC-GGUF`)
   ever had `llama-perplexity` run against it? If yes, the number
   should be in the model card. The current card says "TBD" — is
   there an unpublished number somewhere?

2. Is the padded-centroid bug from `RESEARCH-PLAN.md` §0.1 actually
   fixed? The plan lists it as a prerequisite. I see no
   "fix complete" entry anywhere.

3. The two `FABQ_RC_GGUF_SPEC.md` files disagree. Which one is
   canonical?

4. Do you want me to:
   (a) Write the §9 Step 1-2 fixes (spec number + canonical GGUF) and
       stop there until you have an eval number?
   (b) Write a stripped-down eval harness that downloads the published
       model, runs `compute_perplexity()` on WikiText-2, and dumps the
       result to `results/2026-06-15_qwen27b_fabqrc.json` so we have
       a real baseline?
   (c) Something else?

My recommendation is (a) immediately, then (b) on a GPU box, then we
re-evaluate the micro-scaling proposal with real numbers.
