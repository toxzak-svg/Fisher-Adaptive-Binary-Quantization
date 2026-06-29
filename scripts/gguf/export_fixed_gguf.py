#!/usr/bin/env python3
"""
FABQ-RC GGUF Export (Fixed)
===========================

This script properly exports FABQ-RC quantized weights to GGUF using the
standard llama.cpp pipeline:
1. Reconstruct FP16 weights from FABQ-RC state
2. Write FP16 GGUF via convert.py (or pure-Python)
3. Quantize to Q4_K_M via llama-quantize.exe

This avoids the custom Q1_K packing that was causing the corruption.
"""
import os
import sys
import struct
import json
import gc
import time
import subprocess
import numpy as np
import torch

# Constants
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32
Q4_K_M_ELEMS = 128
Q4_K_M_SIZE = 176

# GGML types
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_K_M = 15

# GGUF types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

MAX_STRING_LEN = 1024 * 1024 * 1024  # 1GB

LLAMA_CPP_DIR = r"C:\Users\Zwmar\AppData\Local\Temp\llama.cpp"
LLAMA_CLI_DIR = r"C:\Users\Zwmar\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"


def align_to(x, a):
    return ((x + a - 1) // a) * a


class SimpleGGUFWriter:
    """Simplified GGUF writer for FP16 weights (for conversion to Q4_K_M)."""

    def __init__(self, path):
        self.path = path
        self.tensors = []
        self.kv = []

    def add_string(self, key, val):
        if isinstance(val, str) and len(val) > MAX_STRING_LEN:
            val = val[:MAX_STRING_LEN]
        self.kv.append((key, GGUF_TYPE_STRING, val))

    def add_uint32(self, key, val):
        self.kv.append((key, GGUF_TYPE_UINT32, val))

    def add_float32(self, key, val):
        self.kv.append((key, GGUF_TYPE_FLOAT32, val))

    def add_uint64(self, key, val):
        self.kv.append((key, GGUF_TYPE_UINT64, val))

    def add_bool(self, key, val):
        self.kv.append((key, GGUF_TYPE_BOOL, bool(val)))

    def add_tensor(self, name, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        self.tensors.append((name, data))

    def write_str(self, f, s):
        enc = s.encode('utf-8')
        f.write(struct.pack('<Q', len(enc)))
        f.write(enc)

    def write(self):
        with open(self.path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.kv)))

            # Metadata
            for key, vtype, val in self.kv:
                self.write_str(f, key)
                f.write(struct.pack('<I', vtype))
                if vtype == GGUF_TYPE_STRING:
                    self.write_str(f, val)
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack('<I', val))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack('<f', val))
                elif vtype == GGUF_TYPE_UINT64:
                    f.write(struct.pack('<Q', val))
                elif vtype == GGUF_TYPE_BOOL:
                    f.write(struct.pack('<B', int(bool(val))))

            # Tensor info
            tensor_meta = []
            offset = 0
            for name, data in self.tensors:
                shape = list(data.shape)
                nbytes = data.nbytes
                n_dims = len(shape)
                self.write_str(f, name)
                f.write(struct.pack('<I', n_dims))
                for d in reversed(shape):
                    f.write(struct.pack('<Q', d))
                f.write(struct.pack('<I', GGML_TYPE_F16))
                f.write(struct.pack('<Q', offset))
                tensor_meta.append((name, data, shape, nbytes))
                offset += align_to(nbytes, ALIGN)

            # Padding
            f.write(b'\x00' * (align_to(f.tell(), ALIGN) - f.tell()))

            # Tensor data
            for name, data, shape, nbytes in tensor_meta:
                f.write(b'\x00' * (align_to(f.tell(), ALIGN) - f.tell()))
                f.write(data.astype(np.float16).tobytes())

            f.flush()


def map_to_gguf_name(fabq_name):
    """Map FABQ-RC layer name to GGUF tensor name."""
    name = fabq_name.replace('model.', '')
    parts = name.split('.')

    if 'layers' in parts:
        layer_idx = parts[parts.index('layers') + 1]
        rest = parts[parts.index('layers') + 2:]

        if 'linear_attn' in rest:
            attn_idx = rest.index('linear_attn')
            proj = rest[attn_idx + 1] if attn_idx + 1 < len(rest) else ''
            proj_map = {
                'q_proj': 'attn_q',
                'k_proj': 'attn_k',
                'v_proj': 'attn_v',
                'out_proj': 'attn_output',
            }
            tensor_type = proj_map.get(proj, proj)
        elif 'mlp' in rest:
            mlp_idx = rest.index('mlp')
            proj = rest[mlp_idx + 1] if mlp_idx + 1 < len(rest) else ''
            proj_map = {
                'gate_proj': 'ffn_gate',
                'up_proj': 'ffn_up',
                'down_proj': 'ffn_down',
            }
            tensor_type = proj_map.get(proj, proj)
        elif 'input_layernorm' in rest:
            tensor_type = 'attn_norm'
        elif 'post_attention_layernorm' in rest:
            tensor_type = 'ffn_norm'
        else:
            tensor_type = '.'.join(rest)

        return f'blk.{layer_idx}.{tensor_type}.weight'

    elif 'lm_head' in parts:
        return 'output.weight'
    elif 'embed_tokens' in parts:
        return 'token_embd.weight'
    elif 'norm' in parts:
        return 'output_norm.weight'

    return name + '.weight'


def reconstruct_fp16_from_fabqrc(state):
    """Reconstruct FP16 weights from FABQ-RC compressed state."""
    layers = state.get('layers', {})

    fp16_tensors = []
    for layer_name, ldata in layers.items():
        out_c = ldata['original_out_features']
        in_c = ldata['original_in_features']
        weight = torch.zeros(out_c, in_c, dtype=torch.float16)

        # Reconstruct int8 channels
        if len(ldata.get('int8_channels', [])) > 0:
            ch = ldata['int8_channels'].long()
            w = ldata['int8_weights'].to(torch.float16)
            s = ldata['int8_scales']
            if s.dim() == 1:
                s = s.unsqueeze(-1)
            weight[ch] = w * s

        # Reconstruct binary channels
        if len(ldata.get('binary_channels', [])) > 0:
            ch = ldata['binary_channels'].long()
            if 'binary_reconstructed_weights' in ldata:
                weight[ch] = ldata['binary_reconstructed_weights']
            else:
                print(f"  WARNING: {layer_name} missing binary_reconstructed_weights")

        fp16_tensors.append((layer_name, weight))

        # Bias
        if ldata.get('bias') is not None:
            fp16_tensors.append((layer_name + '.bias', ldata['bias'].cpu().half()))

    return fp16_tensors


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    print("=" * 60)
    print("FABQ-RC GGUF Export (Fixed)")
    print("=" * 60)

    # Step 1: Download quantized model from HF
    print("\n[1] Downloading FABQ-RC quantized state...")
    HF_TOKEN = os.environ.get('HF_TOKEN', '')
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get('HF_TOKEN') or HF_TOKEN
    except Exception:
        pass

    try:
        model_dir = snapshot_download(
            'toxzak/Qwen3.6-27B-FABQ-RC',
            allow_patterns=['quantized_model.pth'],
            token=HF_TOKEN or None,
        )
        pth_path = os.path.join(model_dir, 'quantized_model.pth')
        size_gb = os.path.getsize(pth_path) / 1e9
        print(f"  Downloaded: {size_gb:.1f} GB")
    except Exception as e:
        print(f"  Download failed: {e}")
        return 1

    # Step 2: Load and reconstruct FP16
    print("\n[2] Loading and reconstructing FP16 weights...")
    t0 = time.time()
    state = torch.load(pth_path, map_location='cpu', weights_only=False)
    t1 = time.time()
    print(f"  Loaded {len(state.get('layers', {}))} layers in {t1-t0:.0f}s")

    version = state.get('version', 'unknown')
    print(f"  Format version: {version}")

    if 'layers' in state:
        fp16_tensors = reconstruct_fp16_from_fabqrc(state)
        print(f"  Reconstructed {len(fp16_tensors)} tensors")
    else:
        print("  ERROR: Unknown state format (no 'layers' key)")
        return 1

    del state
    gc.collect()

    # Step 3: Download base model config and tokenizer
    print("\n[3] Downloading config and tokenizer...")
    BASE_MODEL = 'Qwen/Qwen3.6-27B'

    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    tc = config.to_dict()
    if 'text_config' in tc:
        tc = tc['text_config']

    hpp = tc.get('hidden_size', config.hidden_size)
    nhead = tc.get('num_attention_heads', config.num_attention_heads)
    nkv = tc.get('num_key_value_heads', config.num_key_value_heads)
    nlayer = tc.get('num_hidden_layers', config.num_hidden_layers)
    inter = tc.get('intermediate_size', config.intermediate_size)
    vocab = tc.get('vocab_size', config.vocab_size)
    rope = tc.get('rope_theta', 10000.0)
    ctx = tc.get('max_position_embeddings', tc.get('context_length', 32768))

    print(f"  Config: {nlayer} layers, {hpp} hidden, {nhead} heads, {nkv} KV heads")
    print(f"  Vocab: {vocab}, Context: {ctx}")

    # Step 4: Create FP16 GGUF
    print("\n[4] Creating FP16 GGUF...")
    GGUF_PATH = '/content/Qwen3.6-27B-FABQ-RC-FP16.gguf'
    os.makedirs('/content', exist_ok=True)

    writer = SimpleGGUFWriter(GGUF_PATH)

    # Architecture
    writer.add_string('general.architecture', 'qwen2')
    writer.add_string('general.name', 'Qwen3.6-27B-FABQ-RC')
    writer.add_string('general.description', 'FABQ-RC reconstructed to FP16')
    writer.add_uint32('general.quantization_version', 2)
    writer.add_uint32('general.file_type', 1)  # FP16

    writer.add_uint32('qwen2.block_count', nlayer)
    writer.add_uint32('qwen2.embedding_length', hpp)
    writer.add_uint32('qwen2.feed_forward_length', inter)
    writer.add_uint32('qwen2.context_length', ctx)
    writer.add_uint32('qwen2.attention.head_count', nhead)
    writer.add_uint32('qwen2.attention.head_count_kv', nkv)
    writer.add_uint32('qwen2.rope.dimension_count', hpp // nhead)
    writer.add_float32('qwen2.rope.freq_base', rope)
    writer.add_uint32('qwen2.vocab_size', vocab)

    # Tokenizer - store only essential info, not huge strings
    writer.add_string('tokenizer.ggml.model', 'llama')
    writer.add_uint32('tokenizer.ggml.vocab_size', vocab)
    bos_id = int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else 0
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0
    writer.add_uint32('tokenizer.ggml.bos_token_id', bos_id)
    writer.add_uint32('tokenizer.ggml.eos_token_id', eos_id)

    # Tensors
    print(f"  Writing {len(fp16_tensors)} tensors...")
    for name, tensor in fp16_tensors:
        safe_name = name.replace('/', '.')
        gguf_name = map_to_gguf_name(safe_name)
        writer.add_tensor(gguf_name, tensor.cpu().numpy())

    t0 = time.time()
    writer.write()
    t1 = time.time()
    size_gb = os.path.getsize(GGUF_PATH) / 1e9
    print(f"  GGUF written: {size_gb:.2f} GB in {t1-t0:.0f}s")

    # Step 5: Check if we can use llama-quantize
    llama_quantize = os.path.join(LLAMA_CLI_DIR, 'llama-quantize.exe')
    if not os.path.exists(llama_quantize):
        print(f"\n[5] llama-quantize not found at {llama_quantize}")
        print("  Skipping quantization step.")
        print(f"  FP16 GGUF available at: {GGUF_PATH}")
        print("  To quantize, run manually:")
        print(f"    llama-quantize {GGUF_PATH} <output> Q4_K_M")
        return 0

    # Step 6: Quantize to Q4_K_M
    print("\n[6] Quantizing to Q4_K_M...")
    q4_path = GGUF_PATH.replace('-FP16.gguf', '-Q4_K_M.gguf')

    cmd = [llama_quantize, GGUF_PATH, q4_path, 'Q4_K_M']
    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            size_q4 = os.path.getsize(q4_path) / 1e9
            print(f"  Q4_K_M GGUF created: {size_q4:.2f} GB")
        else:
            print(f"  Quantization failed:")
            print(result.stderr[:500])
    except subprocess.TimeoutExpired:
        print("  Quantization timed out")
    except Exception as e:
        print(f"  Quantization error: {e}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())