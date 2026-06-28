# What Would Make FABQ-RC Publishable

**Subtitle:** The method is interesting. The next step is turning a prototype into a result that other people can reproduce and compare.

**Suggested slug:** `fabq-rc-publishable-roadmap`

FABQ-RC has the shape of a publishable idea, but not yet the evidence package of a publishable result.

That distinction matters.

The idea is clear: use importance-aware allocation, adaptive block sizes, and residual correction to make extreme low-bit quantization less destructive. The repo also has real experiments: weight reconstruction, dense baseline runs, simplified FABQ runtime validation, and variable-precision validation.

But a publishable compression paper needs more than an interesting method and a few prototype measurements.

It needs matched baselines, physical storage accounting, end-to-end perplexity, and reproducible artifacts.

This is the roadmap I would use to get FABQ-RC there.

## 1. Stop Reporting Idealized Bits Per Weight

The first fix is conceptual: report physical bpw.

Earlier FABQ-RC text mentioned targets around 1.18 to 1.21 bits per weight. The repository audit found that the implemented format is higher once storage overhead is counted.

For a representative 3840 x 3840 layer with 5% int4 rows and 95% binary rows:

| Blocksize | Approx. storage per layer | bpw excluding global codebook |
|---:|---:|---:|
| 64 | 3.18 MB | 1.73 |
| 128 | 2.86 MB | 1.55 |
| 256 | 2.69 MB | 1.46 |
| 512 | 2.61 MB | 1.42 |

That is still very compressed. But it is not the same claim as 1.18 bpw.

A strong paper should report:

- Logical payload bpw.
- Physical storage bpw.
- Total model file size.
- Peak memory during inference.
- Whether codebooks, scales, row maps, and metadata are included.

That removes ambiguity and makes comparisons fair.

## 2. Make One GGUF Spec Canonical

The repo currently has conflicting GGUF-format documentation.

That is fine during exploration. It is not fine for a paper or a model release.

Before publishing a strong external claim, there should be one canonical storage spec:

- One tensor layout.
- One metadata scheme.
- One codebook representation.
- One dequantization algorithm.
- One statement of what llama.cpp compatibility means.

If the native compressed format is not fully supported yet, say that directly. If the current benchmark dequantizes back into dense weights, say that too.

Readers will forgive an honest prototype. They will not forgive a format claim that they cannot reproduce.

## 3. Run The Full Fisher Path

FABQ-RC is named after Fisher-adaptive quantization, but the current reported experiments mostly use proxies.

That is acceptable for local prototyping. It is not enough for the final paper.

The paper-grade experiment should run:

```text
calibration data -> forward/backward pass -> Fisher row scores -> allocation -> quantization -> evaluation
```

Then compare it against:

- Row-energy allocation.
- Activation-imatrix allocation.
- Random row allocation.
- Magnitude-based allocation.

This isolates whether Fisher is actually doing useful work.

If Fisher wins, the method earns its name. If it does not, the method should be renamed or reframed around the signal that actually works.

## 4. Implement The Real Residual Codebook

The residual codebook is the most interesting part of the method that is not yet fully represented in the runtime results.

The current simplified benchmark has no full residual codebook. The variable-precision prototype uses residual mean correction for int2 and binary rows. That is not the same as a learned tiered codebook.

The paper-grade version should test:

| Variant | Purpose |
|---|---|
| No residual correction | Baseline |
| Per-block residual mean | Cheap correction |
| Global residual codebook | Tests clustering value |
| Fisher-tiered residual codebooks | Full FABQ-RC idea |

This would answer the core question:

> Does the residual codebook preserve model quality enough to justify its storage and implementation complexity?

Without that ablation, the codebook remains a plausible idea rather than a measured contribution.

## 5. Use Matched Baselines

The current weight reconstruction comparison is useful, but the publishable benchmark needs end-to-end matched baselines.

At minimum:

- Dense FP16 or BF16 baseline.
- Fixed Q1 block baselines.
- int4 rowwise baseline.
- GPTQ-style 4-bit baseline.
- AWQ-style 4-bit baseline.
- BiLLM-style or published BiLLM numbers where comparable.

The key phrase is **matched physical bpw**.

If FABQ-RC uses 1.55 physical bpw, it should be compared to other methods at the closest available physical storage point. If the variable-precision prototype uses 4.53 bpw, it should be compared to normal 4-bit methods and maybe 5-bit methods, not to a theoretical 1-bit target.

This is where a lot of quantization papers get slippery. FABQ-RC should not.

## 6. Replace Smoke Perplexity With Full Evaluation

The current checked-in perplexity runs use small slices, often 256 tokens. That is enough to catch catastrophic failure. It is not enough to claim model quality.

The next evaluation should include:

- WikiText-2 full test perplexity.
- C4 perplexity or a comparable web-text validation set.
- A few downstream tasks such as ARC, HellaSwag, and TruthfulQA or MMLU subsets.
- Generation samples with fixed prompts.
- A small robustness sweep over calibration size.

For each run, save the raw result JSON.

The target is not just a better table. The target is a result someone else can rerun.

## 7. Measure Native Compressed Inference

The current runtime experiments dequantize weights back into dense tensors for validation.

That is a reasonable step. It tells us whether the transformed weights can still support forward passes and generation.

But it does not prove memory savings or speedups during actual compressed inference.

A publishable systems result needs:

- Native compressed matrix multiplication or a clear streaming-dequant path.
- Peak RAM and VRAM measurements.
- Tokens per second.
- Prompt processing speed.
- Decode speed.
- Load time.
- Model file size on disk.

If native compressed inference is not ready, the paper can still be a quantization-quality paper, but it should not sell itself as an inference-speed paper.

## 8. Keep The Negative Result

One thing I would not remove is the failed FABQ-RC-lite experiment.

It is valuable.

It says:

```text
5% int4 + 95% binary + simple importance proxy is not enough.
```

That result justifies the complexity of the real method. It explains why Fisher calibration, residual codebooks, and variable precision are not decorative additions.

If the final paper only includes the best version, readers will not see why the method evolved. If it includes the failure, the paper becomes more credible.

## 9. The Strongest Near-Term Claim

Based on the current repo evidence, I would not lead with:

> FABQ-RC is a 1-bit method that beats BiLLM.

That is not measured.

I would lead with:

> FABQ-RC investigates whether importance-aware allocation and residual correction can make near-binary LLM quantization practical. Early experiments show improved weight reconstruction over fixed binary blocks, a clear failure mode for overly aggressive binary allocation, and a promising variable-precision path that recovers quality at higher physical bpw.

That is less flashy. It is also much harder to attack.

The next target claim could be:

> At matched physical storage, Fisher-tiered residual codebooks improve perplexity over fixed binary block quantization.

That is specific, measurable, and directly connected to the method.

## 10. The Minimal Publishable Experiment

If I had to reduce the roadmap to one paper-grade experiment, it would be this:

1. Pick one small causal model, such as Qwen3-0.6B.
2. Run dense baseline perplexity on full WikiText-2.
3. Quantize with fixed Q1 block64, block128, block256, and block512.
4. Quantize with FABQ-RC-lite.
5. Quantize with full Fisher + residual codebook FABQ-RC.
6. Report physical bpw, file size, MSE, SQNR, and full perplexity.
7. Add ablations for Fisher vs row-energy and codebook vs no-codebook.

If the full FABQ-RC version beats fixed Q1 baselines at matched physical bpw, that is a real result.

After that, scale to 2B, 7B, and 12B.

Only then should the project return to the 27B GGUF claim.

## Takeaway

FABQ-RC does not need more hype. It needs cleaner evidence.

The method is interesting because it asks the right question: how much adaptivity does binary quantization need before it stops destroying language models?

The current experiments show that the naive answer fails. The variable-precision results show a path forward. The storage audit shows what has to be cleaned up before the numbers are credible.

That is enough for a strong research story, but only if the next phase is brutally empirical.

The publishable version of FABQ-RC is not the one with the boldest bpw claim.

It is the one with the cleanest comparison.
