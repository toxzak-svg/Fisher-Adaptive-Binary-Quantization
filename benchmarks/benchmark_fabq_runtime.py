#!/usr/bin/env python3
"""Apply FABQ-style weight quantization, then run runtime validation.

This benchmark quantizes target nn.Linear weights and stores the dequantized
weights back into the Transformers model. That makes perplexity/generation
validation practical on CPU. It does not measure native compressed-kernel
throughput; for that, the CUDA QuantizedLinear path must be built and used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_qwen35_runtime import (  # noqa: E402
    DEFAULT_PROMPT,
    aggregate_loss,
    auto_model_kind,
    load_eval_text,
    run_forward_throughput,
    run_generation,
    run_perplexity,
)


BS_CANDIDATES = (64, 128, 256, 512)


def is_target_linear_name(name: str) -> bool:
    n = name.lower()
    if "embed" in n or "lm_head" in n:
        return False
    if "router" in n or "norm" in n:
        return False
    return True


def ascii_preview(text: str, limit: int = 500) -> str:
    return text[:limit].encode("ascii", errors="backslashreplace").decode("ascii")


def fabq_storage_bits(out_features: int, in_features: int, n_int4: int, blocksize: int) -> int:
    n_binary = out_features - n_int4
    n_blocks = math.ceil(in_features / blocksize)
    int4_bits = n_int4 * in_features * 4
    binary_bits = n_binary * in_features
    binary_scale_bits = n_binary * n_blocks * 16
    int4_scale_bits = n_int4 * 16
    channel_map_bits = out_features * 16
    return int4_bits + binary_bits + binary_scale_bits + int4_scale_bits + channel_map_bits


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def dtype_parameter_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype)
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def _binary_block_error(weight, blocksize: int) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    count = 0
    for start in range(0, weight.shape[1], blocksize):
        block = weight[:, start : start + blocksize]
        scale = block.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        recon = torch.where(block >= 0, scale, -scale)
        diff = block - recon
        total += (diff.double() * diff.double()).sum()
        count += block.numel()
    return float(total / max(count, 1))


def _dequantize_fabq_weight(weight, int4_fraction: float):
    import torch

    w = weight.detach().float().cpu()
    out_features, in_features = w.shape
    n_int4 = max(1, int(round(out_features * int4_fraction)))
    n_int4 = min(n_int4, out_features)
    row_energy = (w * w).mean(dim=1)
    order = torch.argsort(row_energy, descending=True)
    int4_rows = order[:n_int4]
    binary_rows = order[n_int4:]
    recon = torch.empty_like(w)

    if int4_rows.numel():
        iw = w[int4_rows]
        scale = (iw.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-8)
        q = torch.clamp(torch.round(iw / scale), -8, 7)
        recon[int4_rows] = q * scale

    best_bs = 128
    if binary_rows.numel():
        bw = w[binary_rows]
        best_bs = min(BS_CANDIDATES, key=lambda bs: _binary_block_error(bw, bs))
        for start in range(0, in_features, best_bs):
            block = bw[:, start : start + best_bs]
            scale = block.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
            recon[binary_rows, start : start + best_bs] = torch.where(block >= 0, scale, -scale)

    diff = w - recon
    sse = float((diff.double() * diff.double()).sum())
    signal = float((w.double() * w.double()).sum())
    bits = fabq_storage_bits(out_features, in_features, n_int4, best_bs)
    return recon.to(dtype=weight.dtype), {
        "out_features": out_features,
        "in_features": in_features,
        "n_int4": n_int4,
        "n_binary": out_features - n_int4,
        "blocksize": best_bs,
        "sse": sse,
        "signal": signal,
        "weights": out_features * in_features,
        "storage_bits": bits,
    }


def apply_fabq_dequantized(model, int4_fraction: float, max_layers: int = 0) -> dict:
    import torch

    started = time.perf_counter()
    stats = []
    block_hist: dict[str, int] = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not is_target_linear_name(name):
                continue
            if max_layers and len(stats) >= max_layers:
                break
            recon, layer_stats = _dequantize_fabq_weight(module.weight, int4_fraction)
            module.weight.data.copy_(recon.to(device=module.weight.device, dtype=module.weight.dtype))
            layer_stats["name"] = name
            stats.append(layer_stats)
            key = str(layer_stats["blocksize"])
            block_hist[key] = block_hist.get(key, 0) + 1

    total_weights = sum(s["weights"] for s in stats)
    total_sse = sum(s["sse"] for s in stats)
    total_signal = sum(s["signal"] for s in stats)
    total_bits = sum(s["storage_bits"] for s in stats)
    mse = total_sse / max(total_weights, 1)
    signal_mse = total_signal / max(total_weights, 1)
    sqnr_db = 10.0 * math.log10(max(signal_mse, 1e-30) / max(mse, 1e-30))
    return {
        "method": "fabq_dequantized_runtime",
        "method_note": (
            "Target Linear weights are quantized with row-energy importance, "
            "top fraction int4, remaining rows binary, adaptive blocksize, then "
            "dequantized back into dense model weights for CPU perplexity/generation."
        ),
        "layers_quantized": len(stats),
        "target_weights": total_weights,
        "mse": mse,
        "sqnr_db": sqnr_db,
        "estimated_bpw": total_bits / max(total_weights, 1),
        "blocksize_histogram": block_hist,
        "elapsed_sec": time.perf_counter() - started,
        "layers": stats,
    }


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hf-home", default=str(Path.cwd() / ".hf_cache"))
    ap.add_argument("--out", default="results/qwen3_06b_fabq_runtime_benchmark.json")
    ap.add_argument("--max-eval-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--forward-repeats", type=int, default=3)
    ap.add_argument("--dataset-name", default="wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--dataset-max-chars", type=int, default=20000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--int4-fraction", type=float, default=0.05)
    ap.add_argument("--max-layers", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    started = time.perf_counter()
    rss_start = rss_gb()
    config = AutoConfig.from_pretrained(args.repo_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
    model_kind = auto_model_kind(getattr(config, "model_type", None), getattr(config, "architectures", None))
    model_cls = AutoModelForImageTextToText if model_kind == "image_text_to_text" else AutoModelForCausalLM

    load_t0 = time.perf_counter()
    model = model_cls.from_pretrained(
        args.repo_id,
        dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_sec = time.perf_counter() - load_t0
    rss_loaded = rss_gb()

    quantization = apply_fabq_dequantized(model, args.int4_fraction, args.max_layers)
    rss_quantized = rss_gb()

    eval_text, dataset_id = load_eval_text(
        args.dataset_max_chars,
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
    )
    validation = {"loaded": True, "quantized": quantization["layers_quantized"] > 0, "can_forward": False, "can_generate": False}
    ppl = run_perplexity(model, tokenizer, eval_text, args.max_eval_tokens, args.block_size)
    validation["can_forward"] = True
    forward = run_forward_throughput(model, tokenizer, args.prompt, args.forward_repeats)
    generation = run_generation(model, tokenizer, args.prompt, args.max_new_tokens)
    validation["can_generate"] = generation["new_tokens"] > 0

    result = {
        "repo_id": args.repo_id,
        "benchmark_kind": "fabq_dequantized_runtime_validation",
        "architecture": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(getattr(config, "text_config", None), "model_type", None),
        "auto_model_kind": model_kind,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
        },
        "validation": validation,
        "parameter_count": count_parameters(model),
        "dtype_parameter_counts": dtype_parameter_counts(model),
        "rss_gb": {
            "start": rss_start,
            "after_load": rss_loaded,
            "after_fabq_quantize": rss_quantized,
            "after_benchmark": rss_gb(),
        },
        "load_sec": load_sec,
        "dataset": dataset_id,
        "quantization": quantization,
        "perplexity": ppl,
        "forward_throughput": forward,
        "generation": generation,
        "elapsed_sec": time.perf_counter() - started,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    printable = {
        "repo_id": result["repo_id"],
        "validation": validation,
        "quantization": {k: v for k, v in quantization.items() if k != "layers"},
        "perplexity": ppl,
        "forward_throughput": forward,
        "generation": {k: v for k, v in generation.items() if k != "output"},
        "rss_gb": result["rss_gb"],
    }
    print(json.dumps(printable, indent=2))
    print(f"Output preview: {ascii_preview(generation['output'])}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
