# Qwen3-0.6B 1024-Token Baseline and FABQ-VP Check

Date: 2026-06-28

Model: `results/qwen3_06b_bin_checkpoint`

Dataset: `wikitext/wikitext-2-raw-v1/test`

Benchmark kind: local CPU dense baseline and dense-dequantized unified FABQ-VP/EBQ validation. This is not a native compressed-kernel throughput benchmark.

## Runs

| Variant | Estimated bpw | PPL | Scored tokens | PPL forward tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| Dense BF16 | n/a | 50.7062 | 1016 | 17.6632 | 8.9899 |
| Unified FABQ-VP/EBQ target 4.5 | 4.5255 | 59.5981 | 1016 | 19.5803 | 8.7644 |

## Readout

The 4.5255 bpw unified FABQ-VP/EBQ point remains functional on a longer local slice: it loads, quantizes 196 target layers, runs forward perplexity, and generates text.

Quality is still behind dense on this 1024-token WikiText-2 slice. Perplexity increases from 50.7062 to 59.5981, a 17.54% relative increase. That is much better than the near-binary FABQ-RC-lite failure, but it is not yet within the original <5% degradation target.

Throughput is not a meaningful compressed-inference result because the benchmark writes dequantized dense weights back into the model. The forward timing is close to dense, and decode is slightly slower, but this does not prove native packed-runtime speed.

## Artifacts

- Dense baseline JSON: `results/qwen3_06b_dense_baseline_1024.json`
- Unified FABQ-VP/EBQ JSON: `results/qwen3_06b_unified_fabq_bpw45_1024.json`

