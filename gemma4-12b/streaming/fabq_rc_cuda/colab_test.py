# FABQ-RC v2 kernel test on Colab
#
# Run on T4 (free) or L4 (Pro). T4 exercises the full v2 W4A16 fp16-TC
# dispatch path. L4/A100 are faster but the kernel choice is the same:
# we keep activations fp16 because int4xint4 (W4A4) explodes PPL at
# FABQ-RC's 1.21 bpw weight quant.
#
# How to use:
#   1. New Colab notebook (Runtime > Change runtime type > T4 or L4).
#   2. Add secrets in the left-panel key icon: GH_TOKEN, HF_TOKEN.
#   3. Upload this file via the file panel, then run cell-by-cell.
#      (Colab auto-renders .py with # %% markers as cells.)

# %% [markdown]
# # FABQ-RC v2 kernel test
# W4A16 tensor-core path (int4 weight, fp16 act). The rename from
# v2_int4_tc_kernel -> v2_int4_via_fp16_tc_kernel reflects that this
# is fp16x fp16 WMMA, not native int4 TC.

# %% Cell 1: GPU info
!nvidia-smi
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("cc:", torch.cuda.get_device_capability(0))
assert torch.cuda.is_available(), "no GPU, switch runtime to T4 or L4"

# %% Cell 2: deps
!pip install -q pybind11 ninja huggingface_hub pytest

# %% Cell 3: pull tokens, clone repo, auth HF
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

# %% Cell 4: enter the CUDA extension dir
import os
os.chdir('/content/fabq-rc/gemma4-12b/streaming/fabq_rc_cuda')
print("cwd:", os.getcwd())

# %% Cell 5: build the extension (narrow arch list = much faster compile)
import torch
cc_major, cc_minor = torch.cuda.get_device_capability(0)
arch = f"{cc_major}.{cc_minor}"
print(f"Building for sm_{arch} only")
!TORCH_CUDA_ARCH_LIST="{arch}" pip install -e . -q 2>&1 | tail -20

# %% Cell 6: parity tests
!python -m pytest tests/test_v2_kernel.py -v 2>&1 | tail -60

# %% Cell 7: bench - v2 W4A16 fp16-TC vs v1 scalar (dequant-only path)
import torch, time
import sys; sys.path.insert(0, '/content/fabq-rc/gemma4-12b/streaming/fabq_rc_cuda')
from fabq_rc_cuda.quantized_linear import QuantizedLinear

torch.manual_seed(0)
device = 'cuda'

# Build a small int4-only layer so we hit the v2_int4_via_fp16_tc path directly
# (mixed/binary paths have different cost profiles - we'll bench those next).
B, T, IN, OUT = 1, 1, 4096, 4096
n_int4 = OUT  # all rows int4

int4_w = torch.randint(-8, 7, (n_int4, IN), dtype=torch.int8, device=device)
int4_scales = torch.randn(n_int4, dtype=torch.float16, device=device).abs()
row_to_int4 = torch.arange(OUT, dtype=torch.long, device=device)

layer = QuantizedLinear(IN, OUT, bits=4, group_size=128, bias=True).to(device)
# Override the buffers we just made (skip the normal quant path for the bench)
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

# Bench v2 (W4A16 fp16-TC path)
N = 500
t0 = time.time()
for _ in range(N): y = layer(x)
torch.cuda.synchronize()
dt_v2 = (time.time() - t0) / N * 1e3

# Switch to v1 scalar for comparison
layer._use_v2_kernel = False
for _ in range(20): layer(x)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N): y = layer(x)
torch.cuda.synchronize()
dt_v1 = (time.time() - t0) / N * 1e3

print(f"B=1 T=1 {IN}->{OUT} int4 only:")
print(f"  v1 scalar:                       {dt_v1:.3f} ms")
print(f"  v2 W4A16 fp16-TC:               {dt_v2:.3f} ms")
print(f"  speedup:                         {dt_v1/dt_v2:.2f}x")

# %% Cell 8: bench - batched decode (B*T=16, the TC path's target regime)
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
print(f"  v1 scalar:                       {dt_v1_b:.3f} ms")
print(f"  v2 W4A16 fp16-TC:               {dt_v2_b:.3f} ms")
print(f"  speedup:                         {dt_v1_b/dt_v2_b:.2f}x")
