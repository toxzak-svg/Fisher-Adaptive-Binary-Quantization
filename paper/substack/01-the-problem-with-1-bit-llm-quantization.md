# The Problem With 1-Bit LLM Quantization

**Subtitle:** Binary weights are tempting. The hard part is deciding which parts of the model cannot be treated as binary.

**Suggested slug:** `fabq-rc-1-bit-llm-quantization`

Large language models are expensive partly because their weights are huge. If a model has tens of billions of parameters, every bit you shave off the representation matters. Going from FP16 to 4-bit weights is already a major win. Going below that, toward 2-bit or 1-bit weights, is where the idea becomes both exciting and fragile.

The dream is simple: make a model dramatically smaller without retraining it from scratch and without destroying generation quality.

The problem is also simple: language models do not like being treated uniformly.

Some weights can be crushed aggressively and the model barely notices. Other rows, channels, or layers carry much more loss-sensitive behavior. If those get rounded into a crude binary representation, the error propagates through the network and generation falls apart.

FABQ-RC is my attempt to make the binary case less blunt.

FABQ-RC stands for **Fisher-Adaptive Binary Quantization with Residual Codebooks**. The idea is to keep the compression pressure of binary quantization, but spend extra representation capacity exactly where the model appears most sensitive.

The method has four pieces:

1. Estimate which rows or channels matter most.
2. Keep a small critical slice at higher precision.
3. Choose block sizes per layer instead of using one global block size.
4. Learn a residual correction for the systematic errors left behind by binary rounding.

That sounds academic, but the motivation is practical: not every part of a transformer deserves the same number of bits.

## Why Plain Binary Quantization Is Too Blunt

A simple binary quantizer might take a block of weights and represent each value as either `+scale` or `-scale`.

That gives a huge compression win. In the idealized case, each weight needs only one sign bit plus some scale overhead.

But this forces every value in a block into the same two-value shape. If the original block has structure beyond sign and average magnitude, that structure gets erased.

Fixed binary quantization has three obvious weaknesses:

- It treats all rows as equally important.
- It uses the same block size across layers that may have very different distributions.
- It leaves residual errors that are not random noise.

FABQ-RC is built around those three weaknesses.

## The First Bet: Importance Should Drive Precision

The original design uses Fisher information as the importance signal.

In plain terms, Fisher information estimates how much the loss would care if a parameter or row changed. Instead of ranking rows by weight magnitude alone, FABQ-RC tries to rank them by expected loss impact.

The intended score is:

```text
Fisher(row) = expected squared gradient of the loss with respect to that row
```

Rows with high Fisher scores are treated as important. Rows with low Fisher scores are treated as safer to compress.

In the current prototypes, the full Fisher pass is not yet the reported benchmark path. The simpler experiments use cheaper proxies:

- Row energy for the FABQ-RC-lite benchmark.
- Forward activation importance for the variable-precision prototype.

That distinction matters. The method is designed around Fisher, but the currently checked-in empirical results are prototype validations using approximations.

## The Second Bet: Keep a Small High-Precision Core

The simplest FABQ-RC version splits rows into two groups:

| Row group | Representation |
|---|---|
| Most important rows | int4 |
| Remaining rows | binary |

Most experiments use a 5% / 95% split: the top 5% of rows get int4, and the remaining 95% get binary.

This is the core compression gamble.

If the importance ranking is good, a small int4 core should preserve the parts of the layer that matter most. The rest can be pushed into binary form.

If the ranking is bad, or if the model simply cannot tolerate that many binary rows, quality collapses.

That collapse did happen in the simplified experiments. I will cover that in the next post.

## The Third Bet: Block Size Should Be Layer-Specific

Many quantizers use a fixed block size. FABQ-RC instead specifies a per-layer sweep over:

```text
64, 128, 256, 512
```

Smaller blocks usually preserve more local structure but cost more scale overhead. Larger blocks are cheaper but less expressive.

The point is not that one block size is always best. The point is that different layers have different weight distributions, so one global choice is the wrong compromise.

In the repository today, the full adaptive-blocksize claim is only partially validated. The simplified runtime benchmark selected blocksize 64 for all tested layers. The later variable-precision benchmark uses fixed blocksize 128. So the idea is in the design, but the final adaptive sweep still needs stronger end-to-end validation.

## The Fourth Bet: Binary Error Has Structure

After binary quantization, each block has a residual:

```text
residual = original weights - binary reconstruction
```

If that residual were random noise, there would not be much to do. But the hypothesis behind FABQ-RC is that residuals have recurring structure.

The proposed correction is a residual codebook:

1. Quantize the weights.
2. Measure residual blocks.
3. Cluster similar residuals.
4. Store a small codebook of correction patterns.
5. Add the selected correction back during dequantization.

This is where FABQ-RC differs from a purely linear residual approximation. A codebook can represent nonlinear correction patterns. The current CPU benchmark does not yet include the full residual codebook path; it uses either no codebook or a simpler residual mean correction depending on the prototype.

That is one of the biggest open pieces.

## Where This Leaves the Project

The most important thing to say clearly is this:

FABQ-RC is not yet a finished claim that a 1-bit model beats the best existing quantizers.

It is a research prototype with a specific thesis:

> Binary quantization is only plausible if the quantizer is adaptive about importance, block size, and residual error.

The repository already contains enough experiments to show both sides of the story. There is a promising weight-reconstruction result. There is also a hard failure in language-modeling perplexity when the method is too aggressively binary.

That failure is useful. It says the idea cannot be reduced to "5% int4, 95% binary, done." It needs the full calibration and residual correction story, or it needs to become a variable-precision method rather than a pure near-1-bit method.

The next post is about those measurements.

## Takeaway

The interesting part of 1-bit LLM quantization is not whether a weight can be represented as a sign bit. It can.

The interesting part is deciding which weights should not be forced into that representation, and how much correction the binary approximation needs before the model can still behave like a language model.

FABQ-RC is an attempt to answer that question systematically.
