# FABQ-RC Substack Series

This folder contains a three-part Substack adaptation of the FABQ-RC preprint.
The posts are written to be published independently, while still forming a
single narrative arc.

## Suggested Series

### 1. The Problem With 1-Bit LLM Quantization

**File:** `01-the-problem-with-1-bit-llm-quantization.md`

**Angle:** Introduces FABQ-RC through the practical problem: binary quantization
is attractive, but blunt binary rounding breaks language models. This post
explains the intuition behind Fisher-adaptive allocation, adaptive blocksizes,
and residual correction.

**Best audience:** ML engineers, model compression readers, technical founders.

### 2. The Experiment That Failed, And Why It Was Useful

**File:** `02-the-experiment-that-failed.md`

**Angle:** Focuses on the actual measured results. It separates the positive
weight-reconstruction result from the negative perplexity result, then explains
why the variable-precision prototype is the more promising direction.

**Best audience:** Readers who care about benchmarks, failure analysis, and
research credibility.

### 3. What Would Make FABQ-RC Publishable

**File:** `03-what-would-make-fabq-rc-publishable.md`

**Angle:** Turns the current repo state into a concrete research roadmap:
physical bpw, matched baselines, full Fisher calibration, residual codebooks,
and native compressed inference.

**Best audience:** Researchers, collaborators, and people deciding whether the
project is ready to cite, reproduce, or build on.

## Recommended Publication Order

Publish one post at a time. The first post works as the public introduction,
the second builds trust by showing the failure modes, and the third invites
serious follow-up work.

## Tone Notes

- Keep the 1.18 bpw claim framed as unverified historical project context.
- Do not claim FABQ-RC beats BiLLM, GPTQ, AWQ, or Q1 baselines end-to-end yet.
- Lead with measured evidence where possible.
- Treat the current project as an active research prototype, not a finished
  production quantizer.
