#!/usr/bin/env python3
"""Weight-only quantization benchmark for Qwen/Qwen3.5-0.8B.

This is intentionally a tensor-level benchmark. The current repo does not have
an executable Qwen3.5 FABQ-RC inference path, and Qwen3.5 is not a plain
llama.cpp-compatible causal LM architecture. The benchmark therefore compares
the reconstruction/storage behavior of FABQ-RC-style quantization against
nearby weight-only methods on the actual model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


REPO_ID = "Qwen/Qwen3.5-0.8B"
FILENAME = "model.safetensors-00001-of-00001.safetensors"
BS_CANDIDATES = (64, 128, 256, 512)


def is_target_tensor(name: str, shape: tuple[int, ...]) -> bool:
    """Mirror the repo's target-layer policy for FABQ-RC-style layers."""
    if len(shape) != 2:
        return False
    n = name.lower()
    if "embed" in n or "lm_head" in n:
        return False
    if "router" in n:
        return False
    # Existing quant_pipeline.py excludes "gate"; keep MLP gate_proj because
    # it is a normal dense projection in Qwen-family transformer blocks.
    if n.endswith(".bias") or "norm" in n:
        return False
    return True


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.mean(d * d))


def sqnr_db(signal_mse: float, err_mse: float) -> float:
    if err_mse <= 0:
        return float("inf")
    return 10.0 * math.log10(max(signal_mse, 1e-30) / err_mse)


def rowwise_uniform(w: np.ndarray, bits: int) -> tuple[float, float]:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = np.max(np.abs(w), axis=1, keepdims=True) / max(qmax, 1)
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(w / scale), qmin, qmax)
    recon = q * scale
    err = mse(w, recon)
    bpw = bits + (16.0 * w.shape[0] / w.size)
    return err, bpw


def binary_block_recon(w: np.ndarray, blocksize: int) -> tuple[float, float]:
    out, inc = w.shape
    total_sse = 0.0
    for start in range(0, inc, blocksize):
        block = w[:, start : start + blocksize]
        scale = np.mean(np.abs(block), axis=1, keepdims=True)
        scale = np.maximum(scale, 1e-8)
        recon = np.where(block >= 0, scale, -scale)
        diff = block - recon
        total_sse += float(np.sum(diff * diff))
    n_blocks = math.ceil(inc / blocksize) * out
    bpw = 1.0 + (16.0 * n_blocks / w.size)
    return total_sse / w.size, bpw


def weighted_binary_error(w: np.ndarray, row_weight: np.ndarray, blocksize: int) -> float:
    out, inc = w.shape
    weighted_sse = 0.0
    weight_total = 0.0
    rw = row_weight.reshape(out, 1)
    for start in range(0, inc, blocksize):
        block = w[:, start : start + blocksize]
        scale = np.mean(np.abs(block), axis=1, keepdims=True)
        scale = np.maximum(scale, 1e-8)
        recon = np.where(block >= 0, scale, -scale)
        diff2 = (block - recon) ** 2
        weighted_sse += float(np.sum(diff2 * rw))
        weight_total += float(np.sum(rw) * block.shape[1])
    return weighted_sse / max(weight_total, 1e-30)


def fabq_rc_lite(w: np.ndarray, int4_fraction: float) -> tuple[float, float, int]:
    """FABQ-RC-style mix without residual codebook.

    Uses row energy as a local proxy for Fisher channel importance. That keeps
    the benchmark runnable from raw weights while preserving the method shape:
    high-importance output channels are int4; the rest are binary with an
    adaptive per-layer blocksize selected by weighted reconstruction error.
    """
    out, inc = w.shape
    row_importance = np.mean(w * w, axis=1).astype(np.float32)
    row_importance = np.maximum(row_importance, 1e-12)
    n_int4 = max(1, int(round(out * int4_fraction)))
    order = np.argsort(-row_importance)
    int4_rows = order[:n_int4]
    binary_rows = order[n_int4:]

    if binary_rows.size:
        bw = w[binary_rows, :]
        brw = row_importance[binary_rows]
        best_bs = min(BS_CANDIDATES, key=lambda b: weighted_binary_error(bw, brw, b))
    else:
        best_bs = 128

    total_sse = 0.0
    storage_bits = 0.0

    if int4_rows.size:
        iw = w[int4_rows, :]
        err, bpw = rowwise_uniform(iw, 4)
        total_sse += err * iw.size
        storage_bits += bpw * iw.size

    if binary_rows.size:
        bw = w[binary_rows, :]
        err, bpw = binary_block_recon(bw, best_bs)
        total_sse += err * bw.size
        storage_bits += bpw * bw.size

    # Per-output-channel int4/binary membership. uint16 is enough for this model.
    storage_bits += 16.0 * out
    return total_sse / w.size, storage_bits / w.size, best_bs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--filename", default=FILENAME)
    ap.add_argument("--hf-home", default=str(Path.cwd() / ".hf_cache"))
    ap.add_argument("--out", default="results/qwen35_08b_weight_quant.json")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--int4-fraction", type=float, default=0.05)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    t0 = time.time()
    model_path = hf_hub_download(args.repo_id, args.filename)

    methods = ("int8_row", "int4_row", "q1_block64", "q1_block128", "q1_block256", "q1_block512", "fabq_rc_lite")
    totals = {m: {"sse": 0.0, "bits": 0.0} for m in methods}
    signal_sse = 0.0
    total_weights = 0
    tensor_rows = []
    block_hist = Counter()
    skipped = Counter()

    with safe_open(model_path, framework="pt") as f:
        keys = list(f.keys())
        for name in keys:
            shape = tuple(f.get_slice(name).get_shape())
            if not is_target_tensor(name, shape):
                skipped["non_target"] += 1
                continue
            if args.max_tensors and len(tensor_rows) >= args.max_tensors:
                skipped["max_tensors"] += 1
                continue

            tensor = f.get_tensor(name)
            w = tensor.float().cpu().numpy()
            signal = float(np.sum(w * w))
            signal_sse += signal
            total_weights += w.size

            per = {}
            for label, bits in (("int8_row", 8), ("int4_row", 4)):
                err, bpw = rowwise_uniform(w, bits)
                totals[label]["sse"] += err * w.size
                totals[label]["bits"] += bpw * w.size
                per[label] = {"mse": err, "bpw": bpw}

            for bs in BS_CANDIDATES:
                label = f"q1_block{bs}"
                err, bpw = binary_block_recon(w, bs)
                totals[label]["sse"] += err * w.size
                totals[label]["bits"] += bpw * w.size
                per[label] = {"mse": err, "bpw": bpw}

            err, bpw, best_bs = fabq_rc_lite(w, args.int4_fraction)
            totals["fabq_rc_lite"]["sse"] += err * w.size
            totals["fabq_rc_lite"]["bits"] += bpw * w.size
            block_hist[best_bs] += 1
            per["fabq_rc_lite"] = {"mse": err, "bpw": bpw, "blocksize": best_bs}

            tensor_rows.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "weights": int(w.size),
                    "signal_mse": signal / w.size,
                    "methods": per,
                }
            )

    signal_mse = signal_sse / max(total_weights, 1)
    summary = {}
    for method, vals in totals.items():
        err_mse = vals["sse"] / max(total_weights, 1)
        bpw = vals["bits"] / max(total_weights, 1)
        summary[method] = {
            "mse": err_mse,
            "sqnr_db": sqnr_db(signal_mse, err_mse),
            "bpw": bpw,
        }

    result = {
        "repo_id": args.repo_id,
        "model_file": model_path,
        "benchmark_kind": "weight_reconstruction",
        "method_note": (
            "fabq_rc_lite uses row-energy as a Fisher proxy, 5% rowwise int4, "
            "95% binary, adaptive blocksize, and no residual codebook/inference kernel."
        ),
        "target_policy": "2D tensors, excluding embeddings/lm_head/router/norm/bias",
        "elapsed_sec": time.time() - t0,
        "total_target_tensors": len(tensor_rows),
        "total_target_weights": int(total_weights),
        "skipped": dict(skipped),
        "fabq_rc_blocksize_histogram": {str(k): v for k, v in sorted(block_hist.items())},
        "summary": summary,
        "tensors": tensor_rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("benchmark_kind", "elapsed_sec", "total_target_tensors", "total_target_weights", "fabq_rc_blocksize_histogram")}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
