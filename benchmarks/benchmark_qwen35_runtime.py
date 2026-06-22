#!/usr/bin/env python3
"""Validate and benchmark Qwen/Gemma runtime behavior.

The defaults are intentionally small enough to run on a CPU-only workstation:
short WikiText-2 perplexity slice, one short deterministic generation, and a
few forward-pass timing repeats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Iterable

import psutil


DEFAULT_REPO_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "The future of compact language models is"


def auto_model_kind(model_type: str | None, architectures: list[str] | None) -> str:
    arch_text = " ".join(architectures or []).lower()
    model_text = (model_type or "").lower()
    if "conditionalgeneration" in arch_text or model_text in {"qwen3_5", "gemma4", "gemma4_unified"}:
        return "image_text_to_text"
    return "causal_lm"


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def tokens_per_second(prompt_tokens: int, total_tokens: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 0:
        return 0.0
    return max(0, total_tokens - prompt_tokens) / elapsed_sec


def aggregate_loss(loss_token_pairs: Iterable[tuple[float, int]]) -> dict:
    total_loss = 0.0
    total_tokens = 0
    for loss, tokens in loss_token_pairs:
        total_loss += float(loss) * int(tokens)
        total_tokens += int(tokens)
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "tokens": total_tokens,
    }


def load_eval_text(max_chars: int, dataset_name: str, dataset_config: str, split: str) -> tuple[str, str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, dataset_config, split=split)
        pieces: list[str] = []
        total = 0
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            pieces.append(text)
            total += len(text) + 2
            if total >= max_chars:
                break
        if pieces:
            return "\n\n".join(pieces)[:max_chars], f"{dataset_name}/{dataset_config}/{split}"
    except Exception as exc:
        print(f"[warn] dataset load failed, using inline corpus: {type(exc).__name__}: {exc}")

    fallback = (
        "Language models compress patterns in text by predicting the next token. "
        "A useful benchmark should report perplexity, throughput, memory use, "
        "and enough environment detail to make the result reproducible. "
    )
    return (fallback * max(1, max_chars // len(fallback) + 1))[:max_chars], "inline_fallback"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_dtype_summary(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype)
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def run_perplexity(model, tokenizer, text: str, max_tokens: int, block_size: int) -> dict:
    import torch

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][:, :max_tokens]
    if input_ids.size(1) < 2:
        raise ValueError("Need at least two tokens for perplexity")

    pairs: list[tuple[float, int]] = []
    forward_times: list[float] = []
    with torch.inference_mode():
        for start in range(0, input_ids.size(1) - 1, block_size):
            end = min(start + block_size, input_ids.size(1))
            if end - start < 2:
                break
            chunk = input_ids[:, start:end]
            labels = chunk.clone()
            t0 = time.perf_counter()
            out = model(input_ids=chunk, labels=labels)
            elapsed = time.perf_counter() - t0
            valid_tokens = chunk.size(1) - 1
            pairs.append((float(out.loss.detach().cpu()), valid_tokens))
            forward_times.append(elapsed)

    result = aggregate_loss(pairs)
    result["block_size"] = block_size
    result["chunks"] = len(pairs)
    result["eval_tokens"] = int(input_ids.size(1))
    result["forward_sec"] = sum(forward_times)
    result["forward_tokens_per_sec"] = result["tokens"] / max(result["forward_sec"], 1e-12)
    return result


def run_forward_throughput(model, tokenizer, prompt: str, repeats: int) -> dict:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids)
            times.append(time.perf_counter() - t0)
    total_tokens = int(input_ids.numel() * repeats)
    total_sec = sum(times)
    return {
        "prompt_tokens": int(input_ids.numel()),
        "repeats": repeats,
        "total_sec": total_sec,
        "tokens_per_sec": total_tokens / max(total_sec, 1e-12),
        "per_run_sec": times,
    }


def run_generation(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].numel())
    with torch.inference_mode():
        t0 = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t0
    total_tokens = int(output.numel())
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "new_tokens": max(0, total_tokens - prompt_tokens),
        "elapsed_sec": elapsed,
        "new_tokens_per_sec": tokens_per_second(prompt_tokens, total_tokens, elapsed),
        "output": text,
    }


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--hf-home", default=str(Path.cwd() / ".hf_cache"))
    ap.add_argument("--out", default="results/qwen35_08b_runtime_benchmark.json")
    ap.add_argument("--max-eval-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--forward-repeats", type=int, default=3)
    ap.add_argument("--dataset-name", default="wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--dataset-max-chars", type=int, default=20000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
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

    validation = {"loaded": True, "can_forward": False, "can_generate": False}
    ppl = run_perplexity(model, tokenizer, eval_text, args.max_eval_tokens, args.block_size)
    validation["can_forward"] = True
    forward = run_forward_throughput(model, tokenizer, args.prompt, args.forward_repeats)
    generation = run_generation(model, tokenizer, args.prompt, args.max_new_tokens)
    validation["can_generate"] = generation["new_tokens"] > 0

    result = {
        "repo_id": args.repo_id,
        "benchmark_kind": "runtime_validation",
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
        "dtype_parameter_counts": model_dtype_summary(model),
        "rss_gb": {
            "start": rss_start,
            "after_load": rss_loaded,
            "after_benchmark": rss_gb(),
        },
        "load_sec": load_sec,
        "dataset": dataset_id,
        "perplexity": ppl,
        "forward_throughput": forward,
        "generation": generation,
        "elapsed_sec": time.perf_counter() - started,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("repo_id", "validation", "load_sec", "rss_gb", "perplexity", "forward_throughput")}, indent=2))
    print(json.dumps({"generation": {k: v for k, v in generation.items() if k != "output"}}, indent=2))
    print(f"Output preview: {generation['output'][:500]}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
