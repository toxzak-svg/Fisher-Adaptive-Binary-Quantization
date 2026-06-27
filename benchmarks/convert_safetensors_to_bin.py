#!/usr/bin/env python3
"""Convert a local single-shard safetensors checkpoint to pytorch_model.bin.

This intentionally avoids safetensors.safe_open because the current Windows
environment is failing on the memory-map step with OS error 1455.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path


DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def torch_dtype(name: str):
    import torch

    attr = DTYPES.get(name)
    if attr is None:
        raise ValueError(f"unsupported safetensors dtype: {name}")
    return getattr(torch, attr)


def read_safetensors(path: Path) -> dict:
    import torch

    state = {}
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        tensor_items = [(k, v) for k, v in header.items() if k != "__metadata__"]
        for idx, (name, meta) in enumerate(tensor_items, 1):
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            byte_count = end - start
            raw = torch.empty(byte_count, dtype=torch.uint8)
            n_read = f.readinto(raw.numpy())
            if n_read != byte_count:
                raise OSError(f"short read for {name}: {n_read} != {byte_count}")
            tensor = raw.view(torch_dtype(meta["dtype"]))
            state[name] = tensor.reshape(meta["shape"])
            if idx % 50 == 0:
                print(f"loaded {idx}/{len(tensor_items)} tensors")
    return state


def copy_model_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    skip_suffixes = {".safetensors"}
    for path in src_dir.iterdir():
        if path.is_file() and path.suffix not in skip_suffixes:
            shutil.copy2(path, dst_dir / path.name)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    safetensors = list(src_dir.glob("*.safetensors"))
    if len(safetensors) != 1:
        raise ValueError(f"expected one safetensors file in {src_dir}, found {len(safetensors)}")

    copy_model_files(src_dir, out_dir)
    state = read_safetensors(safetensors[0])
    out_file = out_dir / "pytorch_model.bin"
    torch.save(state, out_file)
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
