# Runtime Validation Report

Date: 2026-06-19

Benchmark harness: `benchmarks/benchmark_qwen35_runtime.py`

Environment:

- Python: 3.14.4
- Torch: 2.11.0+cpu
- CUDA: unavailable
- Torch threads: 8
- Dataset: WikiText-2 raw test split
- Eval size: 256 tokens, 2 chunks, block size 128
- Generation: deterministic, 24 new tokens

## Validation Summary

| Model | Auto class | Params | Load | RSS after load | Forward ok | Generate ok |
|---|---|---:|---:|---:|---|---|
| `Qwen/Qwen3.5-0.8B` | image-text-to-text | 852,985,920 | 7.95s | 1.86 GB | yes | yes |
| `Qwen/Qwen3-0.6B` | causal LM | 596,049,920 | 37.77s | 2.09 GB | yes | yes |

Qwen3.5 emitted a Transformers warning that optional fast-path packages for
linear attention are not installed, so its CPU throughput uses the slower Torch
fallback implementation.

## Benchmark Results

| Model | PPL | PPL forward tok/s | Prompt forward tok/s | Decode tok/s | RSS after bench |
|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | 26.5952 | 11.30 | 2.46 | 0.46 | 2.42 GB |
| `Qwen/Qwen3-0.6B` | 35.2165 | 20.96 | 19.06 | 9.96 | 2.20 GB |

## Output Samples

`Qwen/Qwen3.5-0.8B`:

> The future of compact language models is not just about better performance, but about how we can make them more human-like. This is a critical step in the

`Qwen/Qwen3-0.6B`:

> The future of compact language models is promising, but it's also a challenge. The question is, how can we make the future of language models more inclusive

## Readout

`Qwen/Qwen3.5-0.8B` validates functionally: it loads, computes loss, and
generates text. On this CPU machine it is slow, especially for decode, because
its hybrid/multimodal stack does not hit optional optimized linear-attention
kernels.

`Qwen/Qwen3-0.6B` is the better near-term validation target for FABQ-RC runtime
work. It is text-only, simpler to wrap, and roughly 22x faster than Qwen3.5 on
decode in this local CPU benchmark. Its small-slice perplexity is worse here,
but this 256-token run is a smoke benchmark, not a leaderboard-quality eval.

Raw outputs:

- `results/qwen35_08b_runtime_benchmark.json`
- `results/qwen3_06b_runtime_benchmark.json`
