# FABQ Runtime Validation Report

Date: 2026-06-19

Harness: `benchmarks/benchmark_fabq_runtime.py`

Environment:

- Python: 3.14.4
- Torch: 2.11.0+cpu
- CUDA: unavailable
- Dataset: WikiText-2 raw test split
- Eval size: 256 tokens, 2 chunks, block size 128
- Generation: deterministic, 24 new tokens

## Method Tested

This is a CPU validation of the FABQ method shape, not native compressed CUDA
inference. The harness:

1. Loads the dense BF16 Transformers model.
2. Finds target `nn.Linear` layers.
3. Applies row-energy importance as a Fisher proxy.
4. Keeps the top 5% output rows as symmetric int4.
5. Binarizes the remaining rows with an adaptive blocksize from
   `{64, 128, 256, 512}`.
6. Dequantizes the result back into dense model weights.
7. Runs perplexity, forward timing, and generation timing.

This does not include true Fisher calibration or residual codebooks. Those are
critical FABQ-RC components, so this result should be read as a lower-bound
validation of the simplified method variant.

## Dense vs FABQ-Dequantized

| Model | Variant | PPL | Prompt forward tok/s | Decode tok/s | RSS after bench |
|---|---|---:|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | dense BF16 | 26.5952 | 2.46 | 0.46 | 2.42 GB |
| `Qwen/Qwen3.5-0.8B` | FABQ dequantized | 677,505.3533 | 1.93 | 0.42 | 2.77 GB |
| `Qwen/Qwen3-0.6B` | dense BF16 | 35.2165 | 19.06 | 9.96 | 2.20 GB |
| `Qwen/Qwen3-0.6B` | FABQ dequantized | 3,676,448.8825 | 15.29 | 8.79 | 2.22 GB |

## Quantization Stats

| Model | Layers | Target weights | Estimated bpw | SQNR dB | MSE | Blocksize histogram |
|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | 236 | 595,132,416 | 1.4010 | 4.8795 | 6.588762e-05 | `{64: 236}` |
| `Qwen/Qwen3-0.6B` | 196 | 440,401,920 | 1.4004 | 4.8623 | 2.645410e-04 | `{64: 196}` |

## Readout

The simplified FABQ method validates mechanically: both models load, target
layers quantize, forward loss runs, and generation completes.

Quality fails at this setting. Perplexity increases from `26.6` to `677k` on
Qwen3.5-0.8B and from `35.2` to `3.68M` on Qwen3-0.6B. Generated text also
degenerates. The throughput numbers are not meaningful as compressed-inference
evidence because the benchmark stores dequantized dense weights after applying
the method.

The likely missing pieces are the ones the FABQ-RC spec depends on:

- Real calibration Fisher instead of row-energy proxy
- Residual codebook correction
- Possibly a less aggressive binary/int4 split for small models
- Native compressed `QuantizedLinear` runtime for true memory/throughput
  measurement

Raw outputs:

- `results/qwen35_08b_fabq_runtime_benchmark.json`
- `results/qwen3_06b_fabq_runtime_benchmark.json`
