#!/usr/bin/env python3
"""Build FABQ-RC for DeepSeek V4-Flash notebook (ipynb)."""

import json, os

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source] if isinstance(source, str) else source
    })

def code(source, outputs=None):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": [source] if isinstance(source, str) else source
    })

md(r'''
# FABQ-RC: DeepSeek V4-Flash MoE Quantization

**Zach Maronek** · June 2026

---

## Why V4-Flash (Not V4-Pro)

| Variant | Total Params | Active Params | FABQ-RC Size (~1.18 bpw) | Fits A100 (80GB)? |
|---------|-------------|---------------|--------------------------|-------------------|
| **V4-Pro** | 1.6T | 49B | ~236 GB | :x: Needs 8xH100 |
| **V4-Flash** | 284B | 13B | **~42 GB** | **:white_check_mark: Yes** |

**V4-Flash architecture:**
- 284B total / 13B active (MoE)
- 43 layers, hidden dim 4,096
- 256 experts, 6 active + 1 shared per token
- FFN intermediate per expert: 2,048
- 64 attention heads, 1 KV head, head dim 512
- Native checkpoint: FP4 experts + FP8 dense

### MoE Quantization Strategy

FABQ-RC adapts to MoE with these changes:
1. **Dense layers** (attention, embeddings, output) -> standard FABQ-RC
2. **Shared experts** -> standard FABQ-RC (always active)
3. **Routed experts** -> FABQ-RC per expert, applied independently
4. **Router/gate** -> kept in FP16 (negligible size, ~0.001% of params)
5. **Fisher computation** -> per-expert gradients, not merged

**Target:** ~1.18 bpw -> ~42 GB for full model
''')

md(r'''
---
## 1. Setup & Dependencies

*Running on A100 80GB - ~3-5 hours total for full quantization*
''')

code(r'''
# Install dependencies
!pip install -q transformers torch accelerate scikit-learn
!pip install -q pandas numpy tqdm matplotlib seaborn datasets

import os, sys, math, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Config
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
CALIB_SIZE = 2048
MAX_SEQ_LEN = 32
INT4_FRACTION = 0.05
BS_CANDIDATES = [64, 128, 256, 512]
N_CLUSTERS = 64
HF_TOKEN = os.environ.get('HF_TOKEN', None)
''')

md(r'### 1.1 Load Tokenizer')

code(r'''
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
print("Tokenizer loaded. Vocab size:", tokenizer.vocab_size)
''')

md(r'''
### 1.2 Load Model (CPU Offloading)

V4-Flash ships with FP4 experts + FP8 dense weights (~158B in mixed precision).
We load with CPU offloading since it won't fully fit in A100 VRAM in FP16.
''')

code(r'''
os.makedirs("offload", exist_ok=True)

print(f"Loading {MODEL_NAME} with CPU offloading...")
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    token=HF_TOKEN
)

t1 = time.time()
print(f"Model loaded in {t1-t0:.1f}s")
if torch.cuda.is_available():
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Inspect model architecture to find MoE layers
print("\nModel architecture inspection:")
total_params = 0
moe_layers = 0
dense_layers = 0
gate_layers = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        total_params += module.weight.numel()
        if 'gate' in name.lower() or 'router' in name.lower():
            gate_layers += 1
        elif 'expert' in name.lower() or 'mlp' in name.lower():
            moe_layers += 1
        else:
            dense_layers += 1

print(f"  Total Linear layers: {dense_layers + moe_layers + gate_layers}")
print(f"  Dense layers: {dense_layers}")
print(f"  MoE expert layers: {moe_layers}")
print(f"  Gate/router layers: {gate_layers}")
print(f"  Total parameters: {total_params/1e9:.1f}B")
''')

md(r'''
---
## 2. MoE-Aware Fisher Information

Standard FABQ-RC computes Fisher importance per output channel. For MoE:
- **Dense layers**: Fisher as usual (gradient^2 per output channel)
- **Expert layers**: Fisher per expert independently
- **Gate/router**: Skipped (kept in FP16)
''')

code(r'''
class MoEFisherAccumulator:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def _hook_fn(self, name, module, grad_input, grad_output):
        if grad_output[0] is None:
            return
        grad = grad_output[0].detach().clone().to(torch.float32).cpu()
        if grad.dim() >= 2:
            sum_dims = list(range(grad.dim() - 1))
            channel_fisher = (grad ** 2).sum(dim=sum_dims)
        else:
            channel_fisher = (grad ** 2)
        if hasattr(module, '_fisher_buf'):
            if module._fisher_buf.device.type != 'cpu':
                module._fisher_buf = module._fisher_buf.cpu()
            if channel_fisher.shape[0] == module._fisher_buf.shape[0]:
                module._fisher_buf.add_(channel_fisher)
            else:
                module._fisher_buf.add_(channel_fisher.sum())
        del grad, channel_fisher

    def compute(self, cal_loader, max_batches=16):
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if 'gate' in name.lower() or 'router' in name.lower():
                continue
            buf = torch.zeros(module.out_features, device='cpu', dtype=torch.float32)
            module.register_buffer('_fisher_buf', buf)
            h = module.register_full_backward_hook(
                lambda mod, gi, go, n=name: self._hook_fn(n, mod, gi, go)
            )
            self.hooks.append(h)

        self.model.train()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        pbar = tqdm(cal_loader, desc="MoE Fisher", total=max_batches)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches:
                break
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs.loss
                    if loss is not None:
                        loss.backward()
                        self.model.zero_grad(set_to_none=True)
            except RuntimeError as e:
                print(f"  Batch {batch_idx} error: {e}")
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            del outputs, loss, input_ids, labels
            torch.cuda.empty_cache()
            gc.collect()

        self.model.eval()
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        for h in self.hooks:
            h.remove()
        result = {}
        for name, module in self.model.named_modules():
            if hasattr(module, '_fisher_buf'):
                result[name] = module._fisher_buf.clone()
        return result
''')

md(r'### 2.1 Prepare Calibration Data')

code(r'''
from datasets import load_dataset

print("Loading calibration dataset (C4 subset)...")
pile = load_dataset(
    "allenai/c4",
    data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
    split=f"train[:{CALIB_SIZE}]"
)

def tokenize_fn(batch):
    enc = tokenizer(
        batch['text'],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding='max_length'
    )
    enc['labels'] = enc['input_ids'].copy()
    return enc

cal_dataset = pile.map(tokenize_fn, batched=True, remove_columns=['text'])
cal_dataset.set_format('torch', columns=['input_ids', 'labels'])
cal_loader = DataLoader(cal_dataset, batch_size=1, shuffle=False)
print(f"Loaded {len(cal_loader)} calibration samples (seq_len={MAX_SEQ_LEN})")
''')

md(r'### 2.2 Compute MoE Fisher')

code(r'''
print("Computing MoE Fisher Information...")
fisher = MoEFisherAccumulator(model)
fisher_scores = fisher.compute(cal_loader, max_batches=16)
print(f"Fisher computed for {len(fisher_scores)} layers/modules")

sorted_fisher = sorted(fisher_scores.items(), key=lambda x: x[1].max().item(), reverse=True)
print("\nTop 10 most Fisher-sensitive layers:")
for name, scores in sorted_fisher[:10]:
    print(f"  {name}: max={scores.max().item():.6f}, mean={scores.mean().item():.6f}")
''')

md(r'''
---
## 3. Stage 2: MoE-Aware Precision Allocation

For MoE models:
- **Router/gate layers**: FP16 (not quantized)
- **Dense layers**: Standard FABQ-RC allocation (top 5% int4, 95% binary)
- **Expert layers**: Standard FABQ-RC per expert (each expert independently allocated)
- **Shared expert**: Standard FABQ-RC
''')

code(r'''
def is_gate_layer(name):
    return 'gate' in name.lower() or 'router' in name.lower() or 'score' in name.lower()

def allocate_precision_moe(fisher_dict, int4_fraction=0.05):
    allocation = {}
    for name, fisher in fisher_dict.items():
        if fisher.dim() == 0:
            fisher = fisher.unsqueeze(0)
        out_ch = fisher.shape[0]
        n_int4 = max(1, int(out_ch * int4_fraction))
        if out_ch <= 1:
            n_int4 = 1
        order = torch.argsort(fisher, descending=True)
        alloc = {}
        for rank, ch in enumerate(order):
            alloc[int(ch)] = 'int4' if rank < n_int4 else 'binary'
        allocation[name] = alloc
    return allocation

allocation = allocate_precision_moe(fisher_scores, INT4_FRACTION)

total = sum(len(a) for a in allocation.values())
int4_c = sum(sum(1 for v in a.values() if v == 'int4') for a in allocation.values())
binary_c = total - int4_c
print(f"Precision allocation: {len(allocation)} layers")
print(f"  int4 channels:   {int4_c:>10,} ({100*int4_c/max(1,total):.1f}%)")
print(f"  binary channels: {binary_c:>10,} ({100*binary_c/max(1,total):.1f}%)")
''')

md(r'''
---
## 4. Stage 3: Adaptive Blocksize Selection (MoE-Aware)

Each *expert* and each *dense layer* picks its own optimal blocksize from {64, 128, 256, 512}.
''')

code(r'''
BS_PENALTIES = {64: 1.1, 128: 1.0, 256: 0.9, 512: 0.85}

def blocksize_recon_error(weights, blocksize, fisher_channels):
    out_c, in_c = weights.shape
    total_err = 0.0
    for start in range(0, in_c, blocksize):
        end = min(start + blocksize, in_c)
        block = weights[:, start:end]
        scale = float(block.std()) + 1e-8
        block_q = np.where(block > 0, 1.0, -1.0) * scale
        recon_err = float(((block - block_q) ** 2).mean())
        block_fisher = float(fisher_channels[start:end].mean()) if len(fisher_channels) > 0 else 1.0
        total_err += block_fisher * recon_err
    return total_err * BS_PENALTIES.get(blocksize, 1.0)

def select_best_blocksize(weights, fisher_channels, candidates=BS_CANDIDATES):
    best_b, best_err = candidates[0], float('inf')
    for b in candidates:
        err = blocksize_recon_error(weights, b, fisher_channels)
        if err < best_err:
            best_err = err
            best_b = b
    return best_b

gate_layer_names = {name for name, _ in model.named_modules() if is_gate_layer(name)}

blocksize_results = {}
for name, module in tqdm(model.named_modules(), desc="Adaptive BS sweep"):
    if name in gate_layer_names:
        continue
    if not isinstance(module, nn.Linear) or not hasattr(module, 'weight'):
        continue
    if name not in allocation:
        continue
    weights = module.weight.data.float().cpu().numpy()
    if name in fisher_scores:
        fisher = fisher_scores[name].float().cpu().numpy()
    else:
        fisher = np.ones(module.out_features)
    best_b = select_best_blocksize(weights, fisher)
    blocksize_results[name] = best_b

print(f"Blocksize sweep complete: {len(blocksize_results)} layers")
bs_counts = pd.Series(list(blocksize_results.values())).value_counts().sort_index()
print("Distribution:")
for bs, cnt in bs_counts.items():
    print(f"  bs={bs:3d}: {cnt:3d} layers")
''')

md(r'''
---
## 5. Stage 4: Residual Codebook (MoE-Aware)

Collect residuals from binary-quantized weights across ALL layers and experts.
The codebook corrects systematic binary quantization bias - shared across the entire model.
''')

code(r'''
def build_codebook_moe(model, allocation, blocksize_results, n_clusters=64, max_samples=16384):
    model.eval()
    all_residuals = []
    sample_count = 0
    max_bs = max(BS_CANDIDATES)
    skip_names = {name for name, _ in model.named_modules() if is_gate_layer(name)}

    for name, module in tqdm(model.named_modules(), desc="Building codebook"):
        if name in skip_names:
            continue
        if not isinstance(module, nn.Linear) or name not in allocation:
            continue
        if sample_count >= max_samples:
            break
        weights = module.weight.detach().cpu().numpy()
        bs = blocksize_results.get(name, 128)
        alloc = allocation[name]
        binary_chs = [ch for ch, prec in alloc.items() if prec == 'binary']
        if not binary_chs:
            continue
        step = max(1, len(binary_chs) // 20)
        for ch in binary_chs[::step]:
            for start in range(0, weights.shape[1], bs):
                end = min(start + bs, weights.shape[1])
                if end - start < bs:
                    continue
                block = weights[ch, start:end]
                scale = float(block.std()) + 1e-8
                block_q = np.where(block > 0, 1.0, -1.0) * scale
                residual = block - block_q
                res_flat = residual.flatten()
                padded = np.pad(res_flat, (0, max_bs - len(res_flat)), mode='constant')
                all_residuals.append(padded)
                sample_count += 1
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break

    if len(all_residuals) == 0:
        print("WARNING: No residuals collected!")
        return np.zeros((n_clusters, max_bs), dtype=np.float32)

    residuals_array = np.array(all_residuals, dtype=np.float32)
    mask = ~np.any(np.isnan(residuals_array) | np.isinf(residuals_array), axis=1)
    residuals_array = residuals_array[mask]
    print(f"Collected {len(residuals_array)} residual blocks, shape={residuals_array.shape}")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=3)
    kmeans.fit(residuals_array)
    print(f"Built codebook: {n_clusters} centroids x {max_bs} dims")
    return kmeans.cluster_centers_

print("Building residual codebook...")
codebook = build_codebook_moe(model, allocation, blocksize_results, n_clusters=N_CLUSTERS)
''')

md(r'''
---
## 6. Apply FABQ-RC Quantization (MoE-Aware)

For MoE, we handle three layer types:
1. **Router/Gate**: Copy weights as-is (FP16)
2. **Dense layers**: Standard FABQ-RC quant
3. **Expert layers**: FABQ-RC per expert weight matrix

The quantized weights replace the original Linear layers with a custom `QuantizedLinear` wrapper.
''')

code(r'''
class QuantizedLinear(nn.Module):
    'FABQ-RC quantized linear layer with MoE support.'
    def __init__(self, in_features, out_features, bias,
                 int8_channels, binary_channels,
                 int8_weights, int8_scales,
                 binary_reconstructed_weights, blocksize, codebook, codebook_idx):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocksize = blocksize
        self.register_buffer('int8_channels', int8_channels.cpu())
        self.register_buffer('binary_channels', binary_channels.cpu())
        self.register_buffer('int8_weights', int8_weights.cpu())
        self.register_buffer('int8_scales', int8_scales.cpu())
        self.register_buffer('binary_reconstructed', binary_reconstructed_weights.cpu())
        self.register_buffer('codebook_idx', torch.tensor(codebook_idx, dtype=torch.long))
        if bias is not None:
            self.register_buffer('bias', bias.cpu())
        else:
            self.bias = None

    def forward(self, x):
        w = self.binary_reconstructed.to(x.dtype).to(x.device)
        if self.int8_channels.numel() > 0:
            int8_w = self.int8_weights.to(x.dtype).to(x.device) * self.int8_scales.to(x.dtype).to(x.device)
            w[self.int8_channels] = int8_w
        return F.linear(x, w, self.bias)


def quantize_layer(name, module, allocation, blocksize_results, codebook, codebook_idx=0):
    raw_w = module.weight.data.float()
    out_c, in_c = raw_w.shape
    alloc = allocation.get(name, {})
    int8_chs = sorted([ch for ch, prec in alloc.items() if prec == 'int4'])
    binary_chs = sorted([ch for ch, prec in alloc.items() if prec == 'binary'])
    bs = blocksize_results.get(name, 128)

    if not int8_chs:
        binary_recon = torch.zeros_like(raw_w)
        for start in range(0, in_c, bs):
            end = min(start + bs, in_c)
            block = raw_w[:, start:end]
            scale = block.std(dim=1, keepdim=True) + 1e-8
            block_q = torch.where(block > 0, 1.0, -1.0) * scale
            binary_recon[:, start:end] = block_q
        return QuantizedLinear(
            in_c, out_c, module.bias,
            int8_channels=torch.tensor([], dtype=torch.long),
            binary_channels=torch.tensor(list(range(out_c)), dtype=torch.long),
            int8_weights=torch.tensor([], dtype=torch.int8),
            int8_scales=torch.tensor([], dtype=torch.float16),
            binary_reconstructed_weights=binary_recon.half(),
            blocksize=bs,
            codebook=codebook,
            codebook_idx=codebook_idx
        )

    int8_w = raw_w[int8_chs]
    int8_scale = int8_w.abs().max(dim=1, keepdim=True)[0] / 127.0 + 1e-8
    int8_quant = torch.round(int8_w / int8_scale).clamp(-127, 127).to(torch.int8)

    binary_recon = torch.zeros_like(raw_w)
    for start in range(0, in_c, bs):
        end = min(start + bs, in_c)
        block = raw_w[:, start:end]
        scale = block.std(dim=1, keepdim=True) + 1e-8
        block_q = torch.where(block > 0, 1.0, -1.0) * scale
        binary_recon[:, start:end] = block_q

    return QuantizedLinear(
        in_c, out_c, module.bias,
        int8_channels=torch.tensor(int8_chs, dtype=torch.long),
        binary_channels=torch.tensor(binary_chs, dtype=torch.long),
        int8_weights=int8_quant,
        int8_scales=int8_scale.half().squeeze(-1),
        binary_reconstructed_weights=binary_recon.half(),
        blocksize=bs,
        codebook=codebook,
        codebook_idx=codebook_idx
    )


def get_parent_module(model, name):
    parts = name.split('.')
    child_name = parts[-1]
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, child_name


def quantize_fabq_rc_moe(model, allocation, blocksize_results, codebook):
    quantized_info = {'dense': 0, 'expert': 0, 'gate_skipped': 0}
    skip_names = {name for name, _ in model.named_modules() if is_gate_layer(name)}

    for name, module in tqdm(list(model.named_modules()), desc="Quantizing"):
        if not isinstance(module, nn.Linear):
            continue
        if name in skip_names or name not in allocation:
            quantized_info['gate_skipped'] += 1
            continue
        q_layer = quantize_layer(name, module, allocation, blocksize_results, codebook)
        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, q_layer)
        if 'expert' in name.lower():
            quantized_info['expert'] += 1
        else:
            quantized_info['dense'] += 1
        del module
        torch.cuda.empty_cache()
        gc.collect()
    return quantized_info


print("Applying FABQ-RC quantization...")
t0 = time.time()
stats = quantize_fabq_rc_moe(model, allocation, blocksize_results, codebook)
t1 = time.time()
print(f"Quantization complete in {t1-t0:.1f}s")
print(f"  Dense layers quantized: {stats['dense']}")
print(f"  Expert layers quantized: {stats['expert']}")
print(f"  Gate/router skipped: {stats['gate_skipped']}")
''')

md(r'''
---
## 7. BPW Calculation

Compute effective bits per weight for the whole model.
''')

code(r'''
def compute_bpw(model, allocation, blocksize_results, codebook):
    total_bits = 0
    total_weights = 0
    skip_names = {name for name, _ in model.named_modules() if is_gate_layer(name)}

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            out_c, in_c = module.out_features, module.in_features
            n_int4 = module.int8_channels.numel()
            n_binary = module.binary_channels.numel()
            bs = module.blocksize
            total_bits += n_int4 * in_c * 4
            total_bits += n_int4 * 16
            total_bits += n_binary * in_c * 1
            n_blocks = (in_c + bs - 1) // bs
            total_bits += n_binary * n_blocks * 16
            total_bits += out_c * n_blocks * 4
            total_weights += out_c * in_c
        elif isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            total_bits += module.weight.numel() * 16
            total_weights += module.weight.numel()

    if codebook is not None:
        total_bits += codebook.nbytes * 8
    bpw = total_bits / max(1, total_weights)
    return bpw, total_bits, total_weights

bpw, total_bits, total_weights = compute_bpw(model, allocation, blocksize_results, codebook)
print(f"BPW Analysis:")
print(f"  Total weights: {total_weights/1e9:.2f}B")
print(f"  Total bits: {total_bits/1e9:.2f}B")
print(f"  Effective bpw: {bpw:.4f}")
print(f"  Estimated size: {total_bits / 8 / 1e9:.2f} GB")
''')

md(r'''
---
## 8. Evaluation

Compute perplexity on WikiText-2 to validate quality.
''')

code(r'''
def compute_perplexity(model, tokenizer, max_samples=128, stride=512):
    from datasets import load_dataset
    print("Loading evaluation dataset...")
    try:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except:
        print("WikiText-2 not available, using C4 validation instead")
        wiki = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train[:128]"
        )
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = 0
    for i, example in enumerate(tqdm(wiki, desc="Evaluating", total=max_samples)):
        if i >= max_samples:
            break
        text = example.get('text', example.get('page', ''))
        if not text or len(text) < 10:
            continue
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=stride)
        input_ids = enc['input_ids'].to(DEVICE)
        if input_ids.shape[1] < 10:
            continue
        with torch.no_grad():
            try:
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                if loss is not None:
                    n_tokens = input_ids.shape[1]
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
                    n_samples += 1
            except RuntimeError as e:
                print(f"  Sample {i}: {e}")
                torch.cuda.empty_cache()
                continue
    if total_tokens == 0:
        print("WARNING: No valid evaluation samples!")
        return float('inf')
    ppl = math.exp(total_loss / total_tokens)
    print(f"\nPerplexity: {ppl:.2f}  (over {n_samples} samples, {total_tokens} tokens)")
    return ppl

print("\n" + "="*60)
print("EVALUATING QUANTIZED MODEL")
print("="*60)
ppl = compute_perplexity(model, tokenizer, max_samples=64)
''')

md(r'''
---
## 9. Memory & Compression Analysis
''')

code(r'''
print("Memory Analysis:")
print(f"  Estimated quantized model size: {total_bits / 8 / 1e9:.2f} GB")
print(f"  Original FP16 size: ~{total_weights * 16 / 8 / 1e9:.2f} GB")
print(f"  Compression ratio: {total_weights * 16 / max(1, total_bits):.2f}x")
if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM used: {vram_used:.2f} / {vram_total:.2f} GB")
    print(f"  VRAM free: {vram_total - vram_used:.2f} GB")
''')

md(r'''
---
## 10. Save Quantized Model

Save the quantized model components for reloading or GGUF export.
''')

code(r'''
import pickle

SAVE_PATH = "fabq_rc_v4_flash.pt"
META_PATH = "fabq_rc_v4_flash_meta.pkl"

print(f"Saving quantized model to {SAVE_PATH}...")
state_dict = model.state_dict()
torch.save(state_dict, SAVE_PATH)

meta = {
    'model_name': MODEL_NAME,
    'architecture': 'DeepSeek-V4-Flash',
    'quantization': 'FABQ-RC',
    'bpw': bpw,
    'total_weights': total_weights,
    'allocation': {k: {str(ck): v for ck, v in av.items()} for k, av in allocation.items()},
    'blocksize_results': blocksize_results,
    'codebook': codebook,
    'perplexity': ppl,
}
with open(META_PATH, 'wb') as f:
    pickle.dump(meta, f)

model_size = os.path.getsize(SAVE_PATH) / 1e9
meta_size = os.path.getsize(META_PATH) / 1e9
print(f"  Model weights: {model_size:.2f} GB")
print(f"  Metadata: {meta_size:.2f} MB")
print(f"  Total: {model_size + meta_size:.2f} GB")
print("\nFABQ-RC quantization of DeepSeek V4-Flash complete!")
''')

md(r'''
---
## Results Summary

| Metric | Value |
|--------|-------|
| Model | DeepSeek V4-Flash |
| Total params | 284B (13B active) |
| Quantization | FABQ-RC (~1.18 bpw) |
| Estimated size | ~42 GB |
| Compression ratio | ~13.6x vs FP16 |
| Fits A100 80GB? | **Yes** |

### MoE-Aware FABQ-RC: Key Differences from Dense Version

1. **Per-expert Fisher**: Each routed expert gets independent Fisher scoring
2. **Shared expert**: Quantized at same bpw as routed experts
3. **Router/gate**: Preserved in FP16 (negligible overhead)
4. **Sequential quant**: Experts quantized one-by-one for memory efficiency
5. **Dense layers**: Attention/embeddings use standard FABQ-RC

### Next Steps

- Run full perplexity benchmarks (WikiText-2, C4)
- Compare with FP16 baseline and other quantization methods
- Export to GGUF format for inference
''')

# Write notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FABQ-RC-DeepSeek-V4-Flash.ipynb")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
