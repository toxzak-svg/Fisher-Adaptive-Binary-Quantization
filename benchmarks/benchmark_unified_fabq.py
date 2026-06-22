#!/usr/bin/env python3
"""Unified FABQ-VP/EBQ dense-dequantized validation benchmark.

This is a CPU-friendly prototype of the unified spec:
- forward-only imatrix calibration for input-feature importance
- variable precision allocation across int8/int4/int2/binary rows
- residual block correction for int2/binary rows
- dense dequantized weights for perplexity and generation validation

It is not a native compressed-kernel throughput benchmark.
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
from benchmark_fabq_runtime import ascii_preview, is_target_linear_name  # noqa: E402
from benchmark_qwen35_runtime import (  # noqa: E402
    DEFAULT_PROMPT,
    auto_model_kind,
    load_eval_text,
    run_forward_throughput,
    run_generation,
    run_perplexity,
)


BIT_WIDTHS = {"int8": 8, "int4": 4, "int2": 2, "binary": 1}


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


def precision_mix_for_target(target_bpw: float) -> dict[str, float]:
    """Return conservative row fractions for a requested unified-spec budget."""
    if target_bpw <= 2.05:
        return {"int8": 0.02, "int4": 0.18, "int2": 0.30, "binary": 0.50}
    if target_bpw <= 3.05:
        return {"int8": 0.03, "int4": 0.49, "int2": 0.24, "binary": 0.24}
    return {"int8": 0.05, "int4": 0.85, "int2": 0.10, "binary": 0.0}


def estimate_mix_bpw(mix: dict[str, float]) -> float:
    return sum(BIT_WIDTHS[name] * frac for name, frac in mix.items())


def _normalized_importance(input_importance):
    import torch

    imp = input_importance.detach().to(torch.float32).cpu().flatten()
    if imp.numel() == 0:
        return torch.ones(1, dtype=torch.float32)
    imp = torch.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = imp.mean().clamp_min(1e-8)
    return imp / mean


def weighted_mse(weight, recon, input_importance) -> float:
    import torch

    imp = _normalized_importance(input_importance).to(weight.device)
    while imp.numel() < weight.shape[1]:
        imp = torch.cat([imp, imp.new_ones(weight.shape[1] - imp.numel())])
    imp = imp[: weight.shape[1]]
    diff = (weight.to(torch.float32) - recon.to(torch.float32)) ** 2
    return float((diff * imp.view(1, -1)).mean().detach().cpu())


def quantize_symmetric(weight, bits: int, input_importance=None, blocksize: int = 128):
    import torch

    w = weight.detach().to(torch.float32).cpu()
    recon = torch.empty_like(w)
    if input_importance is None:
        imp = torch.ones(w.shape[1], dtype=torch.float32)
    else:
        imp = _normalized_importance(input_importance)
        if imp.numel() < w.shape[1]:
            imp = torch.cat([imp, imp.new_ones(w.shape[1] - imp.numel())])
        imp = imp[: w.shape[1]]

    for start in range(0, w.shape[1], blocksize):
        end = min(start + blocksize, w.shape[1])
        block = w[:, start:end]
        block_imp = imp[start:end].view(1, -1)
        if bits == 1:
            denom = block_imp.sum(dim=1, keepdim=True).clamp_min(1e-8)
            scale = (block.abs() * block_imp).sum(dim=1, keepdim=True) / denom
            recon[:, start:end] = torch.where(block >= 0, scale, -scale)
            continue

        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        denom = max(abs(qmin), abs(qmax))
        scale = (block.abs().amax(dim=1, keepdim=True) / max(denom, 1)).clamp_min(1e-8)
        q = torch.clamp(torch.round(block / scale), qmin, qmax)
        recon[:, start:end] = q * scale
    return recon


def apply_block_residual_correction(weight, recon, blocksize: int = 128):
    import torch

    w = weight.detach().to(torch.float32).cpu()
    out = recon.detach().to(torch.float32).cpu().clone()
    for start in range(0, w.shape[1], blocksize):
        end = min(start + blocksize, w.shape[1])
        residual = w[:, start:end] - out[:, start:end]
        out[:, start:end] += residual.mean(dim=1, keepdim=True)
    return out


def allocate_precision_by_damage(weight, input_importance, mix: dict[str, float]) -> list[str]:
    import torch

    w = weight.detach().to(torch.float32).cpu()
    imp = _normalized_importance(input_importance)
    if imp.numel() < w.shape[1]:
        imp = torch.cat([imp, imp.new_ones(w.shape[1] - imp.numel())])
    imp = imp[: w.shape[1]]
    damage = (w * w * imp.view(1, -1)).mean(dim=1)
    order = torch.argsort(damage, descending=True).tolist()
    n_rows = w.shape[0]
    counts = {name: int(round(frac * n_rows)) for name, frac in mix.items()}
    delta = n_rows - sum(counts.values())
    counts["binary"] = max(0, counts.get("binary", 0) + delta)

    allocation = ["binary"] * n_rows
    cursor = 0
    for name in ("int8", "int4", "int2", "binary"):
        for row in order[cursor : cursor + counts.get(name, 0)]:
            allocation[row] = name
        cursor += counts.get(name, 0)
    return allocation


def storage_bits_for_layer(out_features: int, in_features: int, allocation: list[str], blocksize: int) -> int:
    n_blocks = math.ceil(in_features / blocksize)
    bits = out_features * 16
    for name, width in BIT_WIDTHS.items():
        rows = allocation.count(name)
        if not rows:
            continue
        bits += rows * in_features * width
        scale_blocks = n_blocks if width <= 4 else 1
        bits += rows * scale_blocks * 16
        if width <= 2:
            bits += rows * n_blocks * 16
    return bits


def collect_imatrix(model, tokenizer, text: str, max_tokens: int, block_size: int, max_batches: int) -> dict:
    import torch

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not is_target_linear_name(name):
            continue

        def _hook(mod, inputs, output, layer_name=name):
            if not inputs:
                return
            x = inputs[0].detach().to(torch.float32).cpu()
            if x.shape[-1] != mod.in_features:
                return
            flat = x.reshape(-1, x.shape[-1])
            val = (flat * flat).sum(dim=0)
            if layer_name not in sums:
                sums[layer_name] = val
                counts[layer_name] = flat.shape[0]
            else:
                sums[layer_name] += val
                counts[layer_name] += flat.shape[0]

        hooks.append(module.register_forward_hook(_hook))

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][:, :max_tokens]
    with torch.inference_mode():
        seen = 0
        for start in range(0, input_ids.size(1), block_size):
            if seen >= max_batches:
                break
            chunk = input_ids[:, start : min(start + block_size, input_ids.size(1))]
            if chunk.numel() == 0:
                continue
            _ = model(input_ids=chunk)
            seen += 1

    for h in hooks:
        h.remove()

    return {name: sums[name] / max(counts.get(name, 1), 1) for name in sums}


def _quantize_rows(weight, input_importance, allocation: list[str], blocksize: int):
    import torch

    w = weight.detach().to(torch.float32).cpu()
    recon = torch.empty_like(w)
    for name, bits in BIT_WIDTHS.items():
        rows = torch.tensor([i for i, p in enumerate(allocation) if p == name], dtype=torch.long)
        if rows.numel() == 0:
            continue
        q = quantize_symmetric(w[rows], bits, input_importance, blocksize)
        if bits <= 2:
            q = apply_block_residual_correction(w[rows], q, blocksize)
        recon[rows] = q
    return recon


def apply_unified_dequantized(
    model,
    imatrix: dict,
    target_bpw: float,
    blocksize: int,
    max_layers: int = 0,
) -> dict:
    import torch

    mix = precision_mix_for_target(target_bpw)
    started = time.perf_counter()
    layers = []
    precision_hist = {name: 0 for name in BIT_WIDTHS}
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not is_target_linear_name(name):
                continue
            if max_layers and len(layers) >= max_layers:
                break

            w = module.weight.detach().to(torch.float32).cpu()
            imp = imatrix.get(name)
            if imp is None:
                imp = torch.ones(w.shape[1], dtype=torch.float32)
            allocation = allocate_precision_by_damage(w, imp, mix)
            recon = _quantize_rows(w, imp, allocation, blocksize)
            module.weight.data.copy_(recon.to(device=module.weight.device, dtype=module.weight.dtype))

            diff = w - recon
            sse = float((diff.double() * diff.double()).sum())
            signal = float((w.double() * w.double()).sum())
            wmse = weighted_mse(w, recon, imp)
            layer_bits = storage_bits_for_layer(w.shape[0], w.shape[1], allocation, blocksize)
            for p in allocation:
                precision_hist[p] += 1
            layers.append(
                {
                    "name": name,
                    "out_features": w.shape[0],
                    "in_features": w.shape[1],
                    "weights": w.numel(),
                    "sse": sse,
                    "signal": signal,
                    "weighted_mse": wmse,
                    "storage_bits": layer_bits,
                    "precision_counts": {p: allocation.count(p) for p in BIT_WIDTHS},
                }
            )

    total_weights = sum(layer["weights"] for layer in layers)
    total_sse = sum(layer["sse"] for layer in layers)
    total_signal = sum(layer["signal"] for layer in layers)
    total_bits = sum(layer["storage_bits"] for layer in layers)
    mse = total_sse / max(total_weights, 1)
    signal_mse = total_signal / max(total_weights, 1)
    return {
        "method": "unified_fabq_vp_ebq_dequantized",
        "method_note": (
            "Forward-only imatrix calibration selects variable row precision. "
            "int2/binary rows receive block residual mean correction. Weights are "
            "dequantized back to dense tensors for CPU validation."
        ),
        "target_bpw": target_bpw,
        "mix": mix,
        "mix_nominal_bpw": estimate_mix_bpw(mix),
        "blocksize": blocksize,
        "layers_quantized": len(layers),
        "target_weights": total_weights,
        "mse": mse,
        "sqnr_db": 10.0 * math.log10(max(signal_mse, 1e-30) / max(mse, 1e-30)),
        "estimated_bpw": total_bits / max(total_weights, 1),
        "precision_histogram": precision_hist,
        "elapsed_sec": time.perf_counter() - started,
        "layers": layers,
    }


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hf-home", default=str(Path.cwd() / ".hf_cache"))
    ap.add_argument("--out", default="results/qwen3_06b_unified_fabq_benchmark.json")
    ap.add_argument("--max-eval-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--forward-repeats", type=int, default=3)
    ap.add_argument("--dataset-name", default="wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--dataset-max-chars", type=int, default=20000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--target-bpw", type=float, default=3.0)
    ap.add_argument("--imatrix-tokens", type=int, default=256)
    ap.add_argument("--imatrix-batches", type=int, default=2)
    ap.add_argument("--quant-blocksize", type=int, default=128)
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

    eval_text, dataset_id = load_eval_text(
        args.dataset_max_chars,
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
    )
    imatrix_t0 = time.perf_counter()
    imatrix = collect_imatrix(model, tokenizer, eval_text, args.imatrix_tokens, args.block_size, args.imatrix_batches)
    imatrix_sec = time.perf_counter() - imatrix_t0
    rss_imatrix = rss_gb()

    quantization = apply_unified_dequantized(
        model,
        imatrix,
        target_bpw=args.target_bpw,
        blocksize=args.quant_blocksize,
        max_layers=args.max_layers,
    )
    rss_quantized = rss_gb()

    validation = {"loaded": True, "imatrix_layers": len(imatrix), "quantized": quantization["layers_quantized"] > 0, "can_forward": False, "can_generate": False}
    ppl = run_perplexity(model, tokenizer, eval_text, args.max_eval_tokens, args.block_size)
    validation["can_forward"] = True
    forward = run_forward_throughput(model, tokenizer, args.prompt, args.forward_repeats)
    generation = run_generation(model, tokenizer, args.prompt, args.max_new_tokens)
    validation["can_generate"] = generation["new_tokens"] > 0

    result = {
        "repo_id": args.repo_id,
        "benchmark_kind": "unified_fabq_vp_ebq_dequantized_runtime_validation",
        "architecture": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
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
            "after_imatrix": rss_imatrix,
            "after_unified_quantize": rss_quantized,
            "after_benchmark": rss_gb(),
        },
        "load_sec": load_sec,
        "imatrix_sec": imatrix_sec,
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
