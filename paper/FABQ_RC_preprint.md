---
title: "FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks for Low-Bit LLM Compression"
authors:
  - "Zach Maronek"
date: "2026-06-27"
license: "apache-2.0"
tags:
  - quantization
  - large-language-models
  - low-bit
  - post-training-quantization
  - qwen
  - gguf
---

# FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks for Low-Bit LLM Compression

Zach Maronek

## Abstract

Post-training quantization can reduce the storage and memory cost of large language models (LLMs), but the extreme 1- to 2-bit regime remains brittle. This report introduces FABQ-RC, a Fisher-adaptive binary quantization design that combines channel-wise importance estimation, mixed precision row allocation, per-layer blocksize selection, and residual codebook correction. The central hypothesis is that binary quantization is viable only when the quantizer allocates extra representation capacity to loss-sensitive rows and corrects systematic residual structure after binarization.

We report the current reproducible state of the project rather than a final leaderboard claim. A weight-level benchmark on Qwen3.5-0.8B shows that a simplified FABQ-RC-lite variant improves reconstruction error over fixed binary block quantizers: mean squared error is reduced by 13.3% relative to Q1 block64 and 14.1% relative to Q1 block128, at 1.401 bits per weight. However, dense-dequantized runtime validation shows that the simplified 1.40 bpw variant fails language modeling quality, increasing WikiText-2 slice perplexity from 35.22 to 3.68M on Qwen3-0.6B and from 26.60 to 677k on Qwen3.5-0.8B. A variable-precision prototype using forward activation importance and residual mean correction substantially improves the tradeoff: on Qwen3-0.6B, estimated 4.53 bpw yields 42.50 perplexity on the same 256-token WikiText-2 slice versus 35.22 for the dense baseline, while estimated 3.12 bpw remains inadequate at 3269.77 perplexity.

These results support two conclusions. First, naive binary/int4 mixing is not sufficient; residual correction and less aggressive variable precision are necessary. Second, the previously advertised 1.18 to 1.21 bpw target is not yet supported by the measured storage accounting in this repository. FABQ-RC should therefore be read as an active research prototype with promising reconstruction behavior and a clear validation roadmap, not as a completed 1-bit replacement for production LLM inference.

## 1. Introduction

LLM deployment is frequently limited by weight storage, memory bandwidth, and accelerator memory capacity. Weight-only post-training quantization (PTQ) reduces these costs without retraining the full model. Mature PTQ systems commonly operate in the 3- to 8-bit range, while the binary and near-binary regimes are harder: small quantization errors are amplified across layers, and preserving generation quality requires methods more selective than global blockwise rounding.

FABQ-RC is motivated by three observations.

1. Not all output rows or channels are equally important to loss.
2. Layer weight distributions differ enough that one global blocksize is a poor compromise.
3. Binary quantization leaves structured residuals that can be modeled more flexibly than by a single scale.

The proposed method combines Fisher-weighted importance, a small mixed-precision core, adaptive binary blocksize selection, and residual codebooks. The project also includes a later FABQ-VP/EBQ direction that relaxes the pure 1-bit target into a variable-precision allocation over int8, int4, int2, and binary rows. The experiments in this report evaluate the implemented prototypes and current repository artifacts.

## 2. Related Work

GPTQ introduced a one-shot post-training quantization method using approximate second-order information and demonstrated practical 3- and 4-bit quantization for large transformer models. AWQ showed that preserving or protecting a small set of salient channels, selected from activation statistics, can improve low-bit weight-only quantization. QuIP explored 2-bit quantization with incoherence processing and theoretical guarantees. BiLLM directly targets 1-bit PTQ and combines salient weight selection with binary residual approximation, reporting 1.08-bit LLaMA2-70B with 8.41 perplexity. BitNet b1.58 is adjacent but distinct: it trains ternary networks natively rather than converting an existing dense model through PTQ.

FABQ-RC differs from these directions by combining a Fisher-style loss sensitivity signal with per-layer blocksize adaptation and a nonlinear residual codebook. In the current implementation, full Fisher and residual-codebook evaluation remain incomplete; the empirical results below evaluate simplified and variable-precision prototypes that preserve parts of this design.

## 3. Method

### 3.1 Fisher-Weighted Channel Importance

For a linear layer with weight matrix \(W_l \in R^{d_{out} \times d_{in}}\), FABQ-RC assigns an importance score to each output row. The target score is the diagonal Fisher estimate,

\[
F_{l,i} = E_{(x,y) \sim D}\left[\left\| \frac{\partial L(x,y;\theta)}{\partial W_{l,i,:}} \right\|_2^2\right],
\]

where \(W_{l,i,:}\) is row \(i\) of layer \(l\). Rows with larger Fisher values are expected to cause larger loss changes under quantization noise.

The current benchmark prototypes use cheaper substitutes:

- FABQ-RC-lite uses row energy as a Fisher proxy.
- The unified FABQ-VP/EBQ prototype uses a forward-only input activation matrix, or "imatrix", to weight reconstruction error.

These proxies make local CPU validation feasible but should not be interpreted as a complete Fisher calibration pass.

### 3.2 Mixed Precision Allocation

The original FABQ-RC allocation keeps a small fraction of high-importance rows in int4 and binarizes the remaining rows:

\[
a_{l,i} =
\begin{cases}
\text{int4}, & i \in \text{top-}p \text{ rows by importance} \\
\text{binary}, & \text{otherwise}.
\end{cases}
\]

Most prototype runs use \(p = 0.05\). For binary rows, weights are reconstructed as

\[
\hat{W}_{i,j} = s_{i,b} \cdot \operatorname{sign}(W_{i,j}),
\]

where \(b\) denotes the block containing input coordinate \(j\), and \(s_{i,b}\) is a row-and-block scale.

### 3.3 Adaptive Blocksize Selection

FABQ-RC specifies per-layer selection from block sizes \(\{64, 128, 256, 512\}\). For each candidate blocksize \(B\), the method computes a reconstruction loss weighted by the row importance scores and selects the lowest-error candidate:

\[
B_l^\* = \arg\min_{B \in \{64,128,256,512\}} \sum_i F_{l,i}\|W_{l,i,:} - Q_B(W_{l,i,:})\|_2^2.
\]

The simplified runtime benchmarks selected blocksize 64 for all tested target layers. The unified variable-precision benchmark uses fixed blocksize 128 in the checked-in script, so it is a validation of the allocation idea, not a validation of fully adaptive blocksize search.

### 3.4 Residual Codebook Correction

Binary reconstruction leaves residuals

\[
R_l = W_l - \hat{W}_l.
\]

FABQ-RC proposes tiered residual codebooks. Residual blocks are grouped by Fisher quartile, and each tier receives a learned codebook. During reconstruction, a selected centroid is added back to the binary approximation:

\[
\tilde{W}_{l,b} = \hat{W}_{l,b} + C_{q(l,b), k(l,b)}.
\]

Here \(q(l,b)\) is the Fisher tier for block \(b\), and \(k(l,b)\) is the selected centroid index. The current CPU runtime benchmark does not include this full residual codebook; the unified prototype uses a simpler residual mean correction for int2 and binary rows.

### 3.5 Variable Precision Extension

The later FABQ-VP/EBQ prototype generalizes the two-level scheme to a precision pyramid. Rows are allocated to int8, int4, int2, or binary according to a target average bit budget. The checked-in prototype uses the following mixes:

| Target bpw | int8 | int4 | int2 | binary | Nominal bpw |
|---:|---:|---:|---:|---:|---:|
| 3.0 | 0.03 | 0.49 | 0.24 | 0.24 | 2.92 |
| 4.0 | 0.05 | 0.85 | 0.10 | 0.00 | 4.00 |
| 4.5 | 0.10 | 0.90 | 0.00 | 0.00 | 4.40 |

The implemented storage accounting includes row maps and scale overhead, so estimated bpw is slightly higher than the nominal mix.

## 4. Experimental Setup

The repository contains four relevant experiment classes.

1. Dense runtime baselines for Qwen/Qwen3-0.6B and Qwen/Qwen3.5-0.8B.
2. A weight-only reconstruction benchmark for Qwen/Qwen3.5-0.8B.
3. Dense-dequantized FABQ-RC-lite runtime validation.
4. Dense-dequantized unified FABQ-VP/EBQ runtime validation.

Most local runs used:

- Python 3.14.4
- PyTorch 2.11.0+cpu
- CUDA unavailable
- 8 Torch CPU threads
- WikiText-2 raw test split
- 256 evaluation tokens, two 128-token chunks
- deterministic generation with 24 new tokens

The Qwen3.5-2B unified run was produced in a separate environment with Python 3.12.13, PyTorch 2.11.0+cu128, CUDA available, and an inline fallback corpus because the dataset load failed in offline mode. Its perplexity is therefore not directly comparable to the WikiText-2 runs.

All runtime experiments dequantize weights back into dense tensors before forward evaluation. They validate quantization quality and functional execution, but they do not measure native compressed-kernel speedups.

## 5. Results

### 5.1 Dense Baselines

| Model | Params | Dataset | PPL | PPL forward tok/s | Prompt tok/s | Decode tok/s | RSS after bench |
|---|---:|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | 596,049,920 | WikiText-2 slice | 35.2165 | 20.96 | 19.06 | 9.96 | 2.20 GB |
| Qwen/Qwen3.5-0.8B | 852,985,920 | WikiText-2 slice | 26.5952 | 11.30 | 2.46 | 0.46 | 2.42 GB |

These baselines are small-slice smoke measurements, not full benchmark-suite estimates.

### 5.2 Weight Reconstruction on Qwen3.5-0.8B

The weight-level benchmark covers 244 target tensors and 615,579,648 target weights, excluding embeddings, lm_head, routers, norms, and bias tensors.

| Method | MSE | SQNR dB | bpw |
|---|---:|---:|---:|
| int8 rowwise symmetric | 1.779195e-08 | 40.5900 | 8.0131 |
| int4 rowwise symmetric | 5.767223e-06 | 15.4826 | 4.0131 |
| Q1 block64 | 7.627237e-05 | 4.2685 | 1.2500 |
| Q1 block128 | 7.701788e-05 | 4.2263 | 1.1250 |
| Q1 block256 | 7.751190e-05 | 4.1985 | 1.0625 |
| Q1 block512 | 7.792983e-05 | 4.1752 | 1.0322 |
| FABQ-RC-lite | 6.615134e-05 | 4.8868 | 1.4010 |

FABQ-RC-lite improves MSE over all tested fixed binary block baselines. Relative to Q1 block64, MSE falls by 13.3%. Relative to Q1 block128, MSE falls by 14.1%. The improvement comes with a higher storage budget: 1.401 bpw versus 1.250 bpw for Q1 block64 and 1.125 bpw for Q1 block128.

### 5.3 FABQ-RC-Lite Runtime Validation

| Model | Variant | Estimated bpw | MSE | SQNR dB | PPL | Prompt tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | Dense | n/a | n/a | n/a | 35.2165 | 19.06 | 9.96 |
| Qwen/Qwen3-0.6B | FABQ-RC-lite dequantized | 1.4004 | 2.645410e-04 | 4.8623 | 3,676,448.8825 | 15.29 | 8.79 |
| Qwen/Qwen3.5-0.8B | Dense | n/a | n/a | n/a | 26.5952 | 2.46 | 0.46 |
| Qwen/Qwen3.5-0.8B | FABQ-RC-lite dequantized | 1.4010 | 6.588762e-05 | 4.8795 | 677,505.3533 | 1.93 | 0.42 |

The simplified binary/int4 variant is mechanically valid: models load, quantized weights are substituted, forward passes complete, and generation completes. Quality is not acceptable. This is a negative result and is important: row-energy allocation plus binary block quantization is not enough for usable 1.4 bpw language modeling.

### 5.4 Unified FABQ-VP/EBQ Runtime Validation on Qwen3-0.6B

| Target bpw | Estimated bpw | Mix | MSE | SQNR dB | PPL | Prompt tok/s | Decode tok/s |
|---:|---:|---|---:|---:|---:|---:|---:|
| Dense | n/a | n/a | n/a | n/a | 35.2165 | 19.06 | 9.96 |
| 3.0 | 3.1151 | 3% int8, 49% int4, 24% int2, 24% binary | 8.687487e-05 | 9.6983 | 3269.7708 | 13.70 | 7.49 |
| 4.0 | 4.1432 | 5% int8, 85% int4, 10% int2 | 1.629386e-05 | 16.9670 | 67.4850 | 13.90 | 6.38 |
| 4.5 | 4.5255 | 10% int8, 90% int4 | 7.952114e-06 | 20.0824 | 42.5027 | 14.97 | 8.61 |

The variable-precision prototype strongly outperforms FABQ-RC-lite at comparable evaluation settings. At estimated 4.5255 bpw, the small-slice perplexity gap to dense is 7.2863 absolute points, or approximately 20.7% relative to the dense baseline. At estimated 4.1432 bpw, the gap is still substantial. At estimated 3.1151 bpw, quality remains poor.

### 5.5 Unified Prototype on Qwen3.5-2B

| Model | Target bpw | Estimated bpw | Dataset | MSE | SQNR dB | PPL | Prompt tok/s | Decode tok/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B local checkpoint | 3.0 | 3.1089 | inline fallback | 1.807460e-05 | 9.3759 | 188.5031 | 10.73 | 5.85 |

This run confirms that the prototype scales to a larger checkpoint and validates forward/generation behavior. Because it used an inline fallback corpus rather than WikiText-2, its perplexity should be treated as a functional smoke result only.

### 5.6 Storage Budget Audit

An internal audit found that the previously stated 1.18 to 1.21 bpw target is not supported by the implemented storage accounting. For a representative 3840 x 3840 layer with 5% int4 rows and 95% binary rows, storage is approximately:

| Blocksize | Approx. storage per layer | bpw excluding global codebook |
|---:|---:|---:|
| 64 | 3.18 MB | 1.73 |
| 128 | 2.86 MB | 1.55 |
| 256 | 2.69 MB | 1.46 |
| 512 | 2.61 MB | 1.42 |

The discrepancy appears to come from counting logical payload bits while omitting row maps, scales, block metadata, and other packing overhead. Future claims should report both logical bpw and physical storage bpw.

## 6. Discussion

The results are mixed but informative. FABQ-RC-lite improves tensor reconstruction at near-binary storage, but the language-modeling failure shows that local reconstruction alone is not a sufficient objective in the extreme regime. The unified FABQ-VP/EBQ prototype improves quality by allocating much more of the model to int4 and int8, suggesting that variable precision is a more practical near-term direction than a pure binary target.

The gap between the 3.1 bpw and 4.5 bpw unified runs is especially important. It indicates a sharp quality transition: adding int4 capacity and removing binary rows improves SQNR from 9.70 dB to 20.08 dB and reduces small-slice perplexity from 3269.77 to 42.50. This suggests that the current row allocation and residual mean correction are not expressive enough for high binary fractions.

The storage audit also changes the interpretation of the project. A measured 1.4 to 1.7 bpw physical budget is still highly compressed, but it is not the same claim as 1.18 bpw. Publishable future work should use physical bpw, report end-to-end model file sizes, and evaluate against fixed baselines at matched physical storage.

## 7. Limitations

The current experiments have several major limitations.

1. The full Fisher gradient calibration path is not yet benchmarked in the reported runtime results.
2. The full residual codebook design is specified but not included in the CPU dequantized runtime experiments.
3. Perplexity uses very small 256-token slices and should be treated as smoke validation.
4. Throughput results are for dense-dequantized CPU execution and do not demonstrate compressed-kernel acceleration.
5. The final 27B GGUF claim is not validated by checked-in perplexity logs.
6. The Qwen3.5-2B run used an inline fallback corpus and is not directly comparable to WikiText-2.
7. Existing GGUF specifications in the repository need consolidation before external compatibility claims.

## 8. Reproducibility Artifacts

Relevant repository files:

| File | Role |
|---|---|
| `FABQ_RC_SPEC.md` | Original FABQ-RC method specification |
| `VALIDATION_MEMO.md` | Validation audit and open issues |
| `benchmarks/benchmark_qwen35_runtime.py` | Dense runtime baseline harness |
| `benchmarks/benchmark_qwen35_08b_weight_quant.py` | Weight reconstruction benchmark |
| `benchmarks/benchmark_fabq_runtime.py` | FABQ-RC-lite dense-dequantized runtime harness |
| `benchmarks/benchmark_unified_fabq.py` | Unified FABQ-VP/EBQ dense-dequantized runtime harness |
| `results/qwen35_08b_weight_quant.md` | Weight reconstruction report |
| `results/fabq_runtime_validation_report.md` | Simplified FABQ runtime report |
| `results/runtime_validation_report.md` | Dense baseline runtime report |
| `results/qwen3_06b_unified_fabq*_benchmark.json` | Unified Qwen3-0.6B result files |
| `results/qwen3_5_2b_unified_fabq_benchmark.json` | Unified Qwen3.5-2B result file |

## 9. Future Work

The next publishable milestone is a matched, end-to-end evaluation:

1. Implement and verify the padded-block residual/codebook fix.
2. Run true Fisher calibration rather than row-energy or forward-imatrix proxies.
3. Evaluate the full residual codebook on WikiText-2, C4, and downstream tasks.
4. Compare against Q1 fixed-block, GPTQ, AWQ, and BiLLM-style baselines at matched physical bpw.
5. Build and test the native compressed inference path.
6. Publish exact model sizes, memory usage, perplexity, and generation outputs for each artifact.

## 10. Conclusion

FABQ-RC proposes a structured path toward near-binary post-training quantization: protect important rows, adapt blocksize by layer, and correct residual bias with learned codebooks. Current repository evidence supports the design direction but does not yet support strong claims that the final 1-bit method beats existing baselines. The most reliable positive result is weight-level reconstruction improvement from FABQ-RC-lite over fixed binary quantizers. The most reliable negative result is that this simplified near-binary method fails language modeling quality. The variable-precision prototype recovers quality substantially at 4.5 bpw, suggesting that the strongest near-term research direction is calibrated variable precision with residual correction rather than an immediate pure 1-bit deployment claim.

## References

- Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323, 2022. https://arxiv.org/abs/2210.17323
- Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., and Han, S. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv:2306.00978, 2023. https://arxiv.org/abs/2306.00978
- Chee, J., Cai, Y., Kuleshov, V., and De Sa, C. "QuIP: 2-Bit Quantization of Large Language Models With Guarantees." arXiv:2307.13304, 2023. https://arxiv.org/abs/2307.13304
- Huang, W., Liu, Y., Qin, H., Li, Y., Zhang, S., Liu, X., Magno, M., and Qi, X. "BiLLM: Pushing the Limit of Post-Training Quantization for LLMs." arXiv:2402.04291, 2024. https://arxiv.org/abs/2402.04291
- Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., and Wei, F. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764, 2024. https://arxiv.org/abs/2402.17764
- Yang, A. et al. "Qwen3 Technical Report." arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388
