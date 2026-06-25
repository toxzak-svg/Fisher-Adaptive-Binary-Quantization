"""Build a Colab-ready .ipynb for the FABQ-RC v2 kernel test.

Run from this directory: `python build_colab_notebook.py`.
Output: `FABQ-RC-v2-colab-test.ipynb`.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "FABQ-RC-v2-colab-test.ipynb"

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in text.splitlines()],
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in text.splitlines()],
    })

# -----------------------------------------------------------------------------
md("""# FABQ-RC v2 kernel test

W4A16 tensor-core path (int4 weight, fp16 act). The rename from
`v2_int4_tc_kernel` -> `v2_int4_via_fp16_tc_kernel` reflects that this is
fp16x fp16 WMMA, not native int4 TC. Activations stay fp16 because int4xint4
(W4A4) explodes PPL at FABQ-RC's 1.21 bpw weight quant.

**Before running:** add `GH_TOKEN` and `HF_TOKEN` in the left-panel key icon.
""")

code("""# Cell 1: GPU info
!nvidia-smi
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("cc:", torch.cuda.get_device_capability(0))
assert torch.cuda.is_available(), "no GPU, switch runtime to T4 or L4"
""")

code("""# Cell 2: deps
!pip install -q pybind11 ninja huggingface_hub pytest
""")

code("""# Cell 3: pull tokens, clone repo, auth HF
from google.colab import userdata
import os, subprocess

assert userdata.get('GH_TOKEN'), "add GH_TOKEN to the left-panel key icon"
assert userdata.get('HF_TOKEN'), "add HF_TOKEN to the left-panel key icon"

GH_TOKEN = userdata.get('GH_TOKEN')
HF_TOKEN = userdata.get('HF_TOKEN')
os.environ['GH_TOKEN'] = GH_TOKEN
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HUGGINGFACE_HUB_TOKEN'] = HF_TOKEN   # hf_hub auto-picks this

# Clone shallow + LFS-aware. LFS is enabled on the repo for .gguf + shards.
subprocess.run(['git', 'lfs', 'install'], check=True)
subprocess.run(['git', 'clone', '--depth', '1',
                f'https://{GH_TOKEN}@github.com/toxzak-svg/fabq-rc.git'],
               check=True)

from huggingface_hub import login
login(token=HF_TOKEN)
print("cloned and logged in to HF")
""")

code("""# Cell 4: enter the CUDA extension dir
import os
os.chdir('/content/fabq-rc/gemma4-12b/streaming/fabq_rc_cuda')
print("cwd:", os.getcwd())
""")

code("""# Cell 5: build the extension (narrow arch list = much faster compile)
import torch
cc_major, cc_minor = torch.cuda.get_device_capability(0)
arch = f"{cc_major}.{cc_minor}"
print(f"Building for sm_{arch} only")
!TORCH_CUDA_ARCH_LIST="{arch}" pip install -e . -q 2>&1 | tail -20
""")

code("""# Cell 6: parity tests
!python -m pytest tests/test_v2_kernel.py -v 2>&1 | tail -60
""")

md("""## Bench: v2 (W4A16 fp16-TC) vs v1 (scalar)

Cell 7 hits the decode path (B*T=1, v2_int4_kernel vectorized scalar wins).
Cell 8 hits the batched path (B*T=16, v2_int4_via_fp16_tc_kernel should kick in).
""")

code("""# Cell 7: bench - decode path (B*T=1)
import torch, time
import sys; sys.path.insert(0, '/content/fabq-rc/gemma4-12b/streaming/fabq_rc_cuda')
from fabq_rc_cuda.quantized_linear import QuantizedLinear

torch.manual_seed(0)
device = 'cuda'

# All-int4 layer so we hit the v2_int4_via_fp16_tc path directly.
B, T, IN, OUT = 1, 1, 4096, 4096
n_int4 = OUT

int4_w = torch.randint(-8, 7, (n_int4, IN), dtype=torch.int8, device=device)
int4_scales = torch.randn(n_int4, dtype=torch.float16, device=device).abs()
row_to_int4 = torch.arange(OUT, dtype=torch.long, device=device)

layer = QuantizedLinear(IN, OUT, bits=4, group_size=128, bias=True).to(device)
layer.int4_weights.data = int4_w
layer.int4_scales.data = int4_scales
layer.row_to_int4.data = row_to_int4
layer.int4_channels.data = torch.arange(n_int4, dtype=torch.long, device=device)
layer.binary_bits.data = torch.empty(0, dtype=torch.uint8, device=device)
layer.binary_scales.data = torch.empty(0, 0, dtype=torch.float16, device=device)
layer.codebook_idx.data = torch.empty(0, 0, dtype=torch.uint8, device=device)

x = torch.randn(B, T, IN, device=device, dtype=torch.float16)

# Warmup
for _ in range(20): layer(x)
torch.cuda.synchronize()

N = 500
t0 = time.time()
for _ in range(N): y = layer(x)
torch.cuda.synchronize()
dt_v2 = (time.time() - t0) / N * 1e3

layer._use_v2_kernel = False
for _ in range(20): layer(x)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N): y = layer(x)
torch.cuda.synchronize()
dt_v1 = (time.time() - t0) / N * 1e3

print(f"B=1 T=1 {IN}->{OUT} int4 only:")
print(f"  v1 scalar:                {dt_v1:.3f} ms")
print(f"  v2 W4A16 fp16-TC:         {dt_v2:.3f} ms")
print(f"  speedup:                  {dt_v1/dt_v2:.2f}x")
""")

code("""# Cell 8: bench - batched decode (B*T=16, TC path's target regime)
B, T = 1, 16
x = torch.randn(B, T, IN, device=device, dtype=torch.float16).view(B*T, IN)

layer._use_v2_kernel = True
for _ in range(20): layer(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): y = layer(x)
torch.cuda.synchronize()
dt_v2_b = (time.time() - t0) / N * 1e3

layer._use_v2_kernel = False
for _ in range(20): layer(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): y = layer(x)
torch.cuda.synchronize()
dt_v1_b = (time.time() - t0) / N * 1e3

print(f"B=1 T=16 {IN}->{OUT} int4 only:")
print(f"  v1 scalar:                {dt_v1_b:.3f} ms")
print(f"  v2 W4A16 fp16-TC:         {dt_v2_b:.3f} ms")
print(f"  speedup:                  {dt_v1_b/dt_v2_b:.2f}x")
""")

# -----------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(cells)} cells)")
