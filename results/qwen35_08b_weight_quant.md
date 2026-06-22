# Qwen3.5-0.8B Weight Quantization Benchmark

Date: 2026-06-19

Model: `Qwen/Qwen3.5-0.8B`

Benchmark kind: weight reconstruction over target 2D tensors. This is not a
perplexity or task benchmark. The current repo does not have a runnable
Qwen3.5 FABQ-RC inference path, and Qwen3.5-0.8B is a multimodal hybrid model
(`Qwen3_5ForConditionalGeneration`) rather than a plain llama.cpp-compatible
Qwen causal LM.

Target policy: 2D tensors, excluding embeddings, `lm_head`, routers, norm, and
bias tensors.

Coverage:

- Target tensors: 244
- Target weights: 615,579,648
- Elapsed: 90.56 seconds on CPU
- FABQ-RC-lite selected blocksize histogram: `{64: 244}`

## Aggregate Results

| Method | MSE | SQNR dB | bpw |
|---|---:|---:|---:|
| int8 rowwise symmetric | 1.779195e-08 | 40.5900 | 8.0131 |
| int4 rowwise symmetric | 5.767223e-06 | 15.4826 | 4.0131 |
| Q1 block64 | 7.627237e-05 | 4.2685 | 1.2500 |
| Q1 block128 | 7.701788e-05 | 4.2263 | 1.1250 |
| Q1 block256 | 7.751190e-05 | 4.1985 | 1.0625 |
| Q1 block512 | 7.792983e-05 | 4.1752 | 1.0322 |
| FABQ-RC-lite | 6.615134e-05 | 4.8868 | 1.4010 |

## Interpretation

FABQ-RC-lite improved reconstruction quality over fixed binary methods, but at
materially higher storage:

- vs Q1 block64: 13.3% lower MSE at 1.401 bpw vs 1.250 bpw
- vs Q1 block128: 14.1% lower MSE at 1.401 bpw vs 1.125 bpw
- vs Q1 block512: 15.1% lower MSE at 1.401 bpw vs 1.032 bpw

This benchmark uses row energy as a Fisher proxy, 5% rowwise int4, 95% binary,
adaptive blocksize, and no residual codebook/inference kernel. It should be
treated as a local weight-level sanity benchmark, not as a claim that full
FABQ-RC perplexity has been validated for Qwen3.5-0.8B.

Raw JSON: `results/qwen35_08b_weight_quant.json`
