import json

with open('Main_FABQ_RC_Notebook.ipynb', 'r') as f:
    nb = json.load(f)

# The new cell source (Jupyter cell - so ! shell commands and %magics are fine)
cell_source = """# ================================================================
# FABQ-RC SAVE + GGUF EXPORT + HF UPLOAD
# ================================================================
# Runs on A100 80GB (Colab/Kaggle) — NOT your 16GB local machine
# This cell does THREE things:
#  1. Save FABQ-RC compressed format (~4GB) — for custom loader
#  2. Reconstruct FP16 from QuantizedLinear → convert to GGUF (~3.5GB)
#  3. Upload GGUF to HuggingFace
# ================================================================

import os, torch, bitsandbytes as bnb, shutil
from huggingface_hub import HfApi, create_repo

# Get HF token
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    pass

if not HF_TOKEN:
    raise RuntimeError("Add HF_TOKEN to Colab Secrets with Write permissions")

REPO_ID = "toxzak/Qwen3.6-27B-FABQ-RC"
api = HfApi(token=HF_TOKEN)

# ── STEP 1: Save FABQ-RC compressed format ──────────────────────────
COMPRESSED_PATH = "fabqrc_compressed.pth"
print("[1/3] Saving FABQ-RC compressed format...")
fabqrc_state = save_fabqrc_compressed(
    model, COMPRESSED_PATH, codebook, allocation, blocksize_results
)
print(f"   Saved: {COMPRESSED_PATH}")

# ── STEP 2: Reconstruct FP16 from QuantizedLinear layers ───────────
print("[2/3] Reconstructing FP16 weights from FABQ-RC layers...")

fp16_tensors = {}
quantized_count = 0
for name, module in model.named_modules():
    if 'QuantizedLinear' not in str(type(module)):
        continue
    quantized_count += 1
    out_c = module.original_out_features
    in_c = module.original_in_features
    weight = torch.zeros(out_c, in_c, dtype=torch.float16)

    # Int8 channels: int8_weights * int8_scales
    if module.int8_channels.numel() > 0:
        ch = module.int8_channels.long()
        w = module.int8_weights.to(torch.float16)
        s = module.int8_scales
        weight[ch] = w * s.unsqueeze(-1)

    # Binary channels: pre-reconstructed FP16
    if module.binary_channels.numel() > 0:
        ch = module.binary_channels.long()
        weight[ch] = module.binary_reconstructed_weights

    fp16_tensors[name + '.weight'] = weight.cpu()
    if module.bias is not None:
        fp16_tensors[name + '.bias'] = module.bias.cpu()

print(f"   Reconstructed {quantized_count} QuantizedLinear layers = {len(fp16_tensors)} tensors")

# Save as safetensors (llama.cpp convert.py expects this format)
FP16_DIR = "/tmp/fabqrc_fp16"
shutil.rmtree(FP16_DIR, ignore_errors=True)
os.makedirs(FP16_DIR, exist_ok=True)

try:
    from safetensors.torch import save_file
except ImportError:
    !pip install safetensors -q
    from safetensors.torch import save_file

for i, (name, tensor) in enumerate(fp16_tensors.items()):
    safe_name = name.replace("/", "_").replace(".", "_").replace("-", "_")
    save_file({"weight": tensor}, os.path.join(FP16_DIR, f"layer_{i:04d}.safetensors"))

print(f"   Saved {len(fp16_tensors)} safetensors to {FP16_DIR}")

# ── STEP 3: Convert to GGUF ────────────────────────────────────────
print("[3/3] Installing llama.cpp...")
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp 2>/dev/null || true
!cd /tmp/llama.cpp && git pull origin master 2>/dev/null || true
!cd /tmp/llama.cpp && mkdir -p build && cd build && cmake .. -DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHS="80" 2>&1 | tail -5

GGUF_PATH = "/tmp/Qwen3.6-27B-FABQ-RC-Q1_K_M.gguf"

print("   Converting FP16 safetensors → GGUF Q1_K_M (this takes 10-20 min)...")
!cd /tmp/llama.cpp && python3 convert.py \
    --dir /tmp/fabqrc_fp16 \
    --outfile {GGUF_PATH} \
    --outtype q1_k_m \
    --context 4096 \
    2>&1 | tail -15

if os.path.exists(GGUF_PATH):
    size_gb = os.path.getsize(GGUF_PATH) / 1e9
    print(f"   GGUF ready: {size_gb:.2f} GB")
else:
    print("   GGUF conversion failed — will upload FABQ-RC compressed format instead")
    GGUF_PATH = None

# ── STEP 4: Upload to HuggingFace ──────────────────────────────────
print()
print("[4/4] Uploading to HuggingFace...")
create_repo(repo_id=REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")

if GGUF_PATH and os.path.exists(GGUF_PATH):
    size_gb = os.path.getsize(GGUF_PATH) / 1e9
    print(f"   Uploading GGUF ({size_gb:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=GGUF_PATH,
        path_in_repo="Qwen3.6-27B-FABQ-RC-Q1_K_M.gguf",
        repo_id=REPO_ID, repo_type="model"
    )
    print("   GGUF uploaded!")
else:
    size_gb = os.path.getsize(COMPRESSED_PATH) / 1e9
    print(f"   Uploading FABQ-RC compressed ({size_gb:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=COMPRESSED_PATH,
        path_in_repo="fabqrc_compressed.pth",
        repo_id=REPO_ID, repo_type="model"
    )
    print("   FABQ-RC compressed uploaded!")

# Cleanup
shutil.rmtree(FP16_DIR, ignore_errors=True)
if os.path.exists(COMPRESSED_PATH):
    os.remove(COMPRESSED_PATH)

print()
print("========================================")
print("✅ DONE! Model at: https://huggingface.co/" + REPO_ID)
"""

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [cell_source]
}

# Find and replace the old upload cell (cell 32)
replaced = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'HfApi' in src and 'upload_file' in src and 'torch.save' in src and 'bnb.nn.Linear4bit' in src:
            old_cell = nb['cells'][i]
            print(f"Found upload cell at index {i}")
            print("Old cell preview:")
            print(''.join(old_cell['source'])[:200])
            nb['cells'][i] = new_cell
            replaced = True
            print(f"\nReplaced with new GGUF export cell")
            break

if not replaced:
    print('ERROR: Could not find the old upload cell to replace')
else:
    with open('Main_FABQ_RC_Notebook.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print('Saved successfully')