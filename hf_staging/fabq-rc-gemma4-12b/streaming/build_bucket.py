#!/usr/bin/env python3
"""build_bucket.py - one-time script to populate the FABQ-RC bucket.

What it uploads to `toxzak/gemma-4-12B-it-fabq-rc-bucket`:

  1. The original BF16 shards (mirrored from google/gemma-4-12B-it).
     These are the "source of truth" - if you re-run build_bucket.py
     with a newer calibration, the BF16 stays the same.

  2. The shared FABQ-RC k-means codebook.
     fabqrc-codebook.bin (~256 KB)

  3. FABQ-RC stats: layer_name -> fisher, blocksize, int4/binary allocation.
     fabqrc-stats.json (~50 MB, mostly the per-channel Fisher scores)

  4. Pre-quantized layer shards (the on-disk format from fabq_rc_format.h).
     fabqrc-quantized-00001-of-NNNNN.bin (one per decoder layer, ~25 MB each
     for a 3840x3840 layer; ~25 MB * 48 layers = ~1.2 GB total)

  5. Config + tokenizer + model.safetensors.index.json from the source.

  6. README.md describing the bucket contents.

Usage:
  python build_bucket.py --source google/gemma-4-12B-it --push

If --push is omitted, files are written to ./bucket/ locally and not
uploaded. Use --push to actually upload to the HF bucket.

Hardware: A100 80GB. Total time: ~30-45 min (BF16 download + calibration +
quantization + upload). The BF16 download is the biggest single cost (~24 GB).
"""

import argparse, json, os, sys, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="google/gemma-4-12B-it",
                   help="Source HF model repo (BF16)")
    p.add_argument("--bucket", default="toxzak/gemma-4-12B-it-fabq-rc-bucket",
                   help="Target HF bucket (the FABQ-RC source of truth)")
    p.add_argument("--calib-samples", type=int, default=2048)
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--output-dir", default="./bucket",
                   help="Local staging dir for bucket contents")
    p.add_argument("--push", action="store_true",
                   help="Actually upload to HF (otherwise just stage locally)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--skip-bf16-mirror", action="store_true",
                   help="Don't re-upload the BF16 source shards (use existing)")
    p.add_argument("--skip-upload-bf16", action="store_true",
                   help="Don't re-upload the BF16 source shards even if missing")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== FABQ-RC bucket build ===")
    print(f"Source:  {args.source}")
    print(f"Bucket:  {args.bucket}")
    print(f"Output:  {output_dir.absolute()}")
    print(f"Push:    {args.push}")
    print()

    # --- Step 1: build the fabq_rc_cuda extension ---
    print("[1/7] Building fabq_rc_cuda extension...")
    t0 = time.time()
    ext_dir = Path(__file__).parent / "fabq_rc_cuda"
    os.system(f"cd {ext_dir} && python setup.py build_ext --inplace 2>&1 | tail -10")
    print(f"   {time.time()-t0:.1f}s")
    sys.path.insert(0, str(Path(__file__).parent))
    import fabq_rc_cuda
    print(f"   fabq_rc_cuda loaded (CUDA available: {fabq_rc_cuda.CUDA_AVAILABLE})")
    if not fabq_rc_cuda.CUDA_AVAILABLE:
        print("   WARNING: CUDA extension not available, falling back to PyTorch")

    # --- Step 2: download BF16 source ---
    print(f"\n[2/7] Downloading BF16 source from {args.source}...")
    from huggingface_hub import snapshot_download
    t0 = time.time()
    bf16_path = snapshot_download(
        args.source, token=args.hf_token,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.safetensors",
            "tokenizer*", "preprocessor*", "processor*",
        ],
    )
    print(f"   Downloaded to {bf16_path} in {time.time()-t0:.1f}s")

    # --- Step 3: load model + tokenizer in BF16 ---
    print(f"\n[3/7] Loading {args.source} in BF16 (this takes 1-2 min)...")
    t0 = time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        bf16_path, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(bf16_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Loaded in {time.time()-t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    # --- Step 4: Fisher pass on calibration data ---
    print(f"\n[4/7] Fisher pass ({args.calib_samples} C4 samples, "
          f"seq_len={args.calib_seq_len})...")
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from fabq_rc_cuda import fisher_pass  # see below

    pile = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split=f"train[:{args.calib_samples}]",
    )
    def tokenize_fn(batch):
        enc = tokenizer(batch["text"], truncation=True,
                        max_length=args.calib_seq_len, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc
    cal = pile.map(tokenize_fn, batched=True, remove_columns=["text"])
    cal.set_format("torch", columns=["input_ids", "labels"])
    loader = DataLoader(cal, batch_size=1, shuffle=False)

    t0 = time.time()
    fisher_scores = fisher_pass(model, loader, max_batches=16)
    print(f"   Fisher done for {len(fisher_scores)} layers in {time.time()-t0:.1f}s")

    # --- Step 5: build per-layer blocksize + allocation + codebook ---
    print(f"\n[5/7] Building blocksize + allocation + codebook...")
    t0 = time.time()
    from fabq_rc_cuda.quant_pipeline import (
        select_blocksize_per_layer, allocate_precision, build_codebook,
    )
    blocksize_results = select_blocksize_per_layer(model, fisher_scores)
    allocation = allocate_precision(fisher_scores, int4_fraction=0.05)
    codebook = build_codebook(model, allocation, blocksize_results,
                              n_clusters=64, max_blocksize=512)
    print(f"   {len(blocksize_results)} blocksizes selected, "
          f"codebook shape {tuple(codebook.shape)}, {time.time()-t0:.1f}s")

    # --- Step 6: quantize each layer + write to local files ---
    print(f"\n[6/7] Quantizing layers + writing to {output_dir}...")
    t0 = time.time()
    n_layers = sum(1 for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear)
                   and "embed" not in n.lower()
                   and "lm_head" not in n.lower())
    print(f"   {n_layers} layers to quantize")

    # Determine the per-layer tensor name map
    from safetensors.torch import safe_open, load_file
    index_path = Path(bf16_path) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            wt_index = json.load(f)["weight_map"]
    else:
        wt_index = None

    layer_index_map = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "embed" in name.lower() or "lm_head" in name.lower():
            continue
        layer_idx = int(name.split(".")[2]) if name.count(".") >= 2 else 0
        layer_index_map[name] = layer_idx

    # Quantize layer by layer, write to disk
    from fabq_rc_cuda import _C
    from fabq_rc_cuda.io import save_layer_to_file, save_codebook

    codebook_f32 = codebook.float()
    # Wrap in [1, 64, 512] for the on-disk format
    save_codebook(str(output_dir / "fabqrc-codebook.bin"), codebook.unsqueeze(0))
    print(f"   Wrote fabqrc-codebook.bin "
          f"({os.path.getsize(output_dir / 'fabqrc-codebook.bin') / 1e3:.1f} KB)")

    for layer_name, module in tqdm(
        [(n, m) for n, m in model.named_modules()
         if isinstance(m, torch.nn.Linear)
         and "embed" not in n.lower() and "lm_head" not in n.lower()],
        desc="Quantizing layers",
    ):
        idx = layer_index_map[layer_name]
        alloc = allocation[layer_name]
        bs = blocksize_results.get(layer_name, 128)

        weight = module.weight.data.float().cpu().contiguous()
        int4_chs = torch.tensor(
            sorted([c for c, p in alloc.items() if p == "int4"]),
            dtype=torch.long,
        )
        binary_chs = torch.tensor(
            sorted([c for c, p in alloc.items() if p == "binary"]),
            dtype=torch.long,
        )
        int4_w, int4_s, bin_bits, bin_s, cb_idx = _C.quantize_weight_matrix(
            weight, int4_chs, binary_chs, bs, codebook_f32,
        )
        bias = (module.bias.detach().cpu().to(torch.float16)
                if module.bias is not None else None)
        out_path = output_dir / f"fabqrc-quantized-{idx:05d}.bin"
        save_layer_to_file(
            str(out_path), layer_index=idx,
            in_features=module.in_features, out_features=module.out_features,
            int4_channels=int4_chs, int4_weights=int4_w, int4_scales=int4_s,
            binary_channels=binary_chs, binary_bits=bin_bits,
            binary_scales=bin_s, codebook_idx=cb_idx, blocksize=bs,
            bias=bias,
        )
    print(f"   {time.time()-t0:.1f}s")

    # --- Step 7: write stats + readme + upload ---
    print(f"\n[7/7] Writing stats + README...")
    t0 = time.time()

    stats = {
        "model_source": args.source,
        "n_layers": n_layers,
        "calibration": {
            "dataset": "allenai/c4",
            "n_samples": args.calib_samples,
            "seq_len": args.calib_seq_len,
        },
        "config": {
            "int4_fraction": 0.05,
            "bs_candidates": [64, 128, 256, 512],
            "n_clusters": 64,
            "max_blocksize": 512,
        },
        "layers": {},
    }
    for layer_name, alloc in allocation.items():
        idx = layer_index_map[layer_name]
        stats["layers"][str(idx)] = {
            "name": layer_name,
            "blocksize": blocksize_results[layer_name],
            "n_int4": sum(1 for v in alloc.values() if v == "int4"),
            "n_binary": sum(1 for v in alloc.values() if v == "binary"),
        }
    with open(output_dir / "fabqrc-stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    readme = f"""# {args.bucket}

FABQ-RC quantized version of `{args.source}`. Source of truth for the
streaming notebook in the parent repo.

## Contents

- `fabqrc-codebook.bin` - shared k-means codebook (256 KB)
- `fabqrc-stats.json` - per-layer blocksize + int4/binary allocation
- `fabqrc-quantized-*.bin` - pre-quantized decoder layers (one per layer)
- `*.safetensors` - BF16 source (mirrored from `{args.source}`)

## Layer format

See `../streaming/fabq_rc_cuda/src/fabq_rc_format.h` for the binary layout.

Each layer is:
- int4 channels (5% of channels) at int8 precision with per-row fp16 scale
- binary channels (95%) at 1-bit precision with per-block fp16 scale and
  k-means codebook correction

Total compressed size: ~1.2 GB for 12B model.

## Usage

The streaming notebook in this folder downloads these files, loads the
BF16 source for the tied embedding (~2 GB), and runs inference with the
`fabq_rc_cuda` CUDA extension - no FP16 weight materialization at runtime.
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    print(f"   {time.time()-t0:.1f}s")

    # --- Upload (if --push) ---
    if args.push:
        from huggingface_hub import HfApi
        print(f"\n📤 Uploading to {args.bucket}...")
        api = HfApi(token=args.hf_token)
        api.create_repo(args.bucket, token=args.hf_token,
                        repo_type="model", exist_ok=True)
        for f in output_dir.iterdir():
            if f.is_file():
                print(f"   {f.name} ({f.stat().st_size/1e6:.2f} MB)")
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=args.bucket, repo_type="model",
                )
        print(f"   Done. https://huggingface.co/{args.bucket}")
    else:
        print(f"\n💾 Bucket staged locally at {output_dir}")
        print(f"   Re-run with --push to upload to HF")

    print(f"\n=== Bucket build complete ===")


# Fisher pass — local helper
def fisher_pass(model, loader, max_batches=16):
    """One forward+backward pass over calibration data, accumulate gradient²
    per output channel for every nn.Linear."""
    from tqdm.auto import tqdm
    import torch
    import torch.nn as nn
    import gc

    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "embed" in name.lower() or "lm_head" in name.lower():
            continue
        if "gate" in name.lower() or "router" in name.lower():
            continue
        module.register_buffer(
            "_fisher_buf",
            torch.zeros(module.out_features, device="cpu", dtype=torch.float32),
        )

        def _hook(mod, gi, go, m=module):
            if go[0] is None: return
            grad = go[0].detach().clone().to(torch.float32).cpu()
            if grad.dim() == 3:
                cf = (grad ** 2).sum(dim=[0, 1])
            else:
                cf = (grad ** 2).sum(dim=list(range(grad.dim() - 1)))
            if cf.shape[0] == m._fisher_buf.shape[0]:
                m._fisher_buf.add_(cf)
            del grad, cf

        h = module.register_full_backward_hook(_hook)
        hooks.append(h)

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    device = next(model.parameters()).device

    pbar = tqdm(loader, desc="Fisher", total=max_batches)
    for i, batch in enumerate(pbar):
        if i >= max_batches: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(ids, labels=lbl)
                if out.loss is not None:
                    out.loss.backward()
                    model.zero_grad(set_to_none=True)
        except RuntimeError as e:
            print(f"  Batch {i}: {e}")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        del out, ids, lbl
        torch.cuda.empty_cache()
        gc.collect()

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for h in hooks: h.remove()

    return {n: m._fisher_buf.clone()
            for n, m in model.named_modules() if hasattr(m, "_fisher_buf")}


if __name__ == "__main__":
    from tqdm.auto import tqdm
    import torch
    main()
