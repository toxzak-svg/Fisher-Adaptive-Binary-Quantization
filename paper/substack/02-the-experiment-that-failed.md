# The Experiment That Failed, And Why It Was Useful

**Subtitle:** FABQ-RC-lite improved weight reconstruction, then failed perplexity. That is exactly the kind of result a compression project needs to take seriously.

**Suggested slug:** `fabq-rc-failed-experiment-results`

The first version of a quantization method can look good for the wrong reason.

If you only measure weight reconstruction error, a method can appear promising while still destroying language-modeling quality. If you only measure storage, it can appear compact while hiding scale overhead, row maps, or metadata. If you only report target perplexity, you can end up publishing a hope instead of a result.

That is why the current FABQ-RC results are useful even though some of them are negative.

They force the project to separate three things:

1. Does the quantizer reconstruct weights better than a nearby baseline?
2. Does the quantized model still produce usable language-modeling loss?
3. Does the measured storage match the advertised bits-per-weight?

Right now, the honest answers are: yes on one weight-level benchmark, no for the aggressive binary runtime prototype, and not yet for the older 1.18 bpw claim.

## The Positive Result: Better Weight Reconstruction

The cleanest positive result is a weight reconstruction benchmark on `Qwen/Qwen3.5-0.8B`.

The benchmark covers:

- 244 target tensors.
- 615,579,648 target weights.
- 2D tensors only.
- Embeddings, `lm_head`, routers, norms, and bias tensors excluded.

The simplified method is called **FABQ-RC-lite**. It is not the full FABQ-RC design. It uses row energy as a Fisher proxy, keeps 5% of rows in int4, binarizes the rest, selects a blocksize, and does not include the full residual codebook.

Here are the aggregate reconstruction results:

| Method | MSE | SQNR dB | bpw |
|---|---:|---:|---:|
| int8 rowwise symmetric | 1.779195e-08 | 40.5900 | 8.0131 |
| int4 rowwise symmetric | 5.767223e-06 | 15.4826 | 4.0131 |
| Q1 block64 | 7.627237e-05 | 4.2685 | 1.2500 |
| Q1 block128 | 7.701788e-05 | 4.2263 | 1.1250 |
| Q1 block256 | 7.751190e-05 | 4.1985 | 1.0625 |
| Q1 block512 | 7.792983e-05 | 4.1752 | 1.0322 |
| FABQ-RC-lite | 6.615134e-05 | 4.8868 | 1.4010 |

FABQ-RC-lite improves reconstruction error over the fixed binary baselines. Relative to Q1 block64, MSE is 13.3% lower. Relative to Q1 block128, MSE is 14.1% lower.

That is real signal.

It also comes at a real cost. FABQ-RC-lite uses 1.401 bits per weight in this accounting, compared with 1.250 bpw for Q1 block64 and 1.125 bpw for Q1 block128.

So the result is not "free quality." It is a trade: better reconstruction for more storage.

## The Negative Result: Perplexity Collapsed

Weight reconstruction is not enough.

The next benchmark took the simplified FABQ-RC-lite quantized weights, dequantized them back into dense tensors, and ran language-modeling validation. This does not measure compressed-kernel speed. It measures whether the model can still run forward and generate after the weight transformation.

It could run.

It could not preserve quality.

| Model | Variant | Estimated bpw | PPL |
|---|---|---:|---:|
| Qwen/Qwen3-0.6B | Dense | n/a | 35.2165 |
| Qwen/Qwen3-0.6B | FABQ-RC-lite dequantized | 1.4004 | 3,676,448.8825 |
| Qwen/Qwen3.5-0.8B | Dense | n/a | 26.5952 |
| Qwen/Qwen3.5-0.8B | FABQ-RC-lite dequantized | 1.4010 | 677,505.3533 |

That is not a small degradation. That is model failure.

The important point is that this failure happened even though the reconstruction benchmark improved over fixed Q1 baselines.

This is the difference between "the tensor looks closer" and "the model still works."

For LLM quantization, that distinction is everything.

## Why The Failure Matters

The failure narrows the search space.

The simplified recipe was:

```text
5% int4 rows
95% binary rows
row-energy importance proxy
adaptive binary blocksize
no full residual codebook
```

That recipe is too aggressive, at least in the tested form.

There are several likely reasons:

- Row energy is too weak as an importance proxy.
- 95% binary rows is too much for these small models.
- Local reconstruction error does not capture downstream sensitivity.
- The missing residual codebook is not optional.
- Small models may be less forgiving than larger models at extreme bit widths.

The conclusion is not "FABQ-RC is dead." The conclusion is more specific:

> A near-binary FABQ-RC-lite shortcut is not enough.

That is a useful result because it prevents the project from overstating the method before the hard parts are implemented.

## The Variable-Precision Prototype Looked Better

After that failure, the more promising direction became the unified FABQ-VP/EBQ prototype.

Instead of forcing almost everything into binary, this version allocates rows across int8, int4, int2, and binary. It uses forward activation importance and a simple residual mean correction for int2 and binary rows.

On `Qwen/Qwen3-0.6B`, the results looked like this:

| Target bpw | Estimated bpw | Mix | MSE | SQNR dB | PPL |
|---:|---:|---|---:|---:|---:|
| Dense | n/a | n/a | n/a | n/a | 35.2165 |
| 3.0 | 3.1151 | 3% int8, 49% int4, 24% int2, 24% binary | 8.687487e-05 | 9.6983 | 3269.7708 |
| 4.0 | 4.1432 | 5% int8, 85% int4, 10% int2 | 1.629386e-05 | 16.9670 | 67.4850 |
| 4.5 | 4.5255 | 10% int8, 90% int4 | 7.952114e-06 | 20.0824 | 42.5027 |

This is still not a final benchmark. It is a 256-token WikiText-2 slice, and the model is dequantized back to dense CPU weights for validation.

But it shows the direction clearly.

At 3.115 bpw, quality is still poor. At 4.143 bpw, the model is much closer but still meaningfully worse than dense. At 4.525 bpw, the model gets to 42.50 perplexity versus 35.22 for dense on the same small slice.

That is not a victory lap. But it is a recovery.

It says the method becomes much more viable when it stops pretending every row can survive binary compression.

## The Storage Audit Changed The Story Too

The repo also contains an important storage audit.

Earlier public-facing project text mentioned targets around 1.18 to 1.21 bpw. The implemented storage accounting does not currently support those numbers.

For a representative 3840 x 3840 layer with 5% int4 rows and 95% binary rows, the physical storage budget is closer to:

| Blocksize | Approx. storage per layer | bpw excluding global codebook |
|---:|---:|---:|
| 64 | 3.18 MB | 1.73 |
| 128 | 2.86 MB | 1.55 |
| 256 | 2.69 MB | 1.46 |
| 512 | 2.61 MB | 1.42 |

The likely issue is that older numbers counted logical payload bits but did not fully account for row maps, scales, block metadata, and packing overhead.

That matters because "1.18 bpw" and "1.55 physical bpw" are different claims.

Future FABQ-RC results need to report physical bpw, not just idealized payload bits.

## What I Take From This

The current evidence supports a more sober version of the project:

- FABQ-RC-lite improves weight reconstruction over fixed binary block baselines.
- That improvement does not transfer to usable perplexity in the aggressive near-binary prototype.
- Variable precision looks much more promising than pure near-1-bit compression.
- The older 1.18 bpw claim should not be repeated as a measured result.

This is not the clean story you want if you are trying to market a finished quantizer.

It is exactly the story you want if you are trying to turn a speculative method into real research.

The failed experiment tells us where the compression cliff is. The variable-precision run tells us one way to back away from it. The storage audit tells us what numbers need to be cleaned up before the method is publishable.

That is progress.

## Takeaway

The most useful result in FABQ-RC so far is not that it "wins."

It is that the project now has a sharper boundary between what works, what fails, and what still needs to be proven.

Near-binary quantization is not dead. But the shortcut version is.
