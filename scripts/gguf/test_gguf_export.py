#!/usr/bin/env python3
"""
FABQ-RC GGUF Export - Local Test with Qwen2.5-0.5B
===================================================

Tests the GGUF export pipeline with a small model before doing the full 27B.
This avoids downloading the 50GB FABQ-RC model until we know the pipeline works.
"""
import os
import sys
import struct
import gc
import time
import subprocess
import numpy as np
import torch

# Constants
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32

# GGML types
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_K_M = 15

# GGUF types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

LLAMA_CLI_DIR = r"C:\Users\Zwmar\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
OUTPUT_DIR = r"C:\Users\Zwmar\AppData\Local\Temp\fabq-test"


def align_to(x, a):
    return ((x + a - 1) // a) * a


class GGUFWriter:
    def __init__(self, path):
        self.path = path
        self.tensors = []
        self.kv = []

    def add_string(self, key, val):
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
            pad = align_to(f.tell(), ALIGN) - f.tell()
            if pad > 0:
                f.write(b'\x00' * pad)

            # Tensor data
            for name, data, shape, nbytes in tensor_meta:
                pad = align_to(f.tell(), ALIGN) - f.tell()
                if pad > 0:
                    f.write(b'\x00' * pad)
                f.write(data.astype(np.float16).tobytes())

            f.flush()


def map_to_gguf_name(name):
    """Simple mapping for Qwen2.5."""
    if 'model.embed_tokens' in name:
        return 'token_embd.weight'
    if 'model.norm' in name:
        return 'output_norm.weight'
    if 'lm_head' in name:
        return 'output.weight'
    # layers.0.mlp.gate_proj.weight -> blk.0.ffn_gate.weight
    parts = name.split('.')
    if 'layers' in parts:
        layer_idx = parts[parts.index('layers') + 1]
        rest = parts[parts.index('layers') + 2:]
        if 'mlp' in rest:
            proj_idx = rest.index('mlp') + 1
            proj = rest[proj_idx] if proj_idx < len(rest) else ''
            proj_map = {'gate_proj': 'ffn_gate', 'up_proj': 'ffn_up', 'down_proj': 'ffn_down'}
            return f'blk.{layer_idx}.{proj_map.get(proj, proj)}.weight'
        if 'linear_attn' in rest or 'self_attn' in rest:
            proj_idx = max(rest.index('linear_attn') if 'linear_attn' in rest else -1,
                          rest.index('self_attn') if 'self_attn' in rest else -1) + 1
            proj = rest[proj_idx] if proj_idx < len(rest) and proj_idx > 0 else ''
            proj_map = {'q_proj': 'attn_q', 'k_proj': 'attn_k', 'v_proj': 'attn_v', 'out_proj': 'attn_output'}
            return f'blk.{layer_idx}.{proj_map.get(proj, proj)}.weight'
        if 'input_layernorm' in rest:
            return f'blk.{layer_idx}.attn_norm.weight'
        if 'post_attention_layernorm' in rest:
            return f'blk.{layer_idx}.ffn_norm.weight'
    return name.replace('/', '.') + '.weight'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FABQ-RC GGUF Export Test (Qwen2.5-0.5B)")
    print("=" * 60)

    # Step 1: Load Qwen2.5-0.5B (memory-efficient loading)
    print("\n[1] Loading Qwen2.5-0.5B...")
    model_path = r"C:\Users\Zwmar\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987\model.safetensors"

    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        print("  Run: huggingface-cli download Qwen/Qwen2.5-0.5B")
        return 1

    print(f"  Loading safetensors from {model_path}")

    # Load tensors one at a time to minimize memory
    state = {}
    tensor_names = []
    try:
        from safetensors import safe_open
        with safe_open(model_path, framework="pt") as f:
            for key in f.keys():
                tensor_names.append(key)
    except ImportError:
        print("  safetensors not available, using torch.load")
        return 1

    print(f"  Found {len(tensor_names)} tensors in safetensors")

    def load_tensor(name):
        with safe_open(model_path, framework="pt") as f:
            return f.get_tensor(name)

    # Step 2: Get config
    print("\n[2] Creating GGUF metadata...")
    config_path = r"C:\Users\Zwmar\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987\config.json"
    import json
    with open(config_path) as f:
        config = json.load(f)

    tc = config.get('text_config', config)
    hpp = tc.get('hidden_size', 896)
    nhead = tc.get('num_attention_heads', 7)
    nkv = tc.get('num_key_value_heads', 2)
    nlayer = tc.get('num_hidden_layers', 24)
    inter = tc.get('intermediate_size', 4864)
    vocab = tc.get('vocab_size', 128256)
    rope = tc.get('rope_theta', 1000000.0)
    ctx = tc.get('max_position_embeddings', 32768)
    rms_eps = tc.get('rms_norm_eps', 1e-6)

    print(f"  Config: {nlayer} layers, {hpp} hidden, {nhead} heads")

    # Step 3: Create FP16 GGUF
    print("\n[3] Creating FP16 GGUF...")
    gguf_path = os.path.join(OUTPUT_DIR, 'Qwen2.5-0.5B-FP16.gguf')

    writer = GGUFWriter(gguf_path)

    # Architecture
    writer.add_string('general.architecture', 'qwen2')
    writer.add_string('general.name', 'Qwen2.5-0.5B-FABQ-test')
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
    writer.add_float32('qwen2.attention.layer_norm_rms_epsilon', rms_eps)

    # Tokenizer
    writer.add_string('tokenizer.ggml.model', 'llama')
    writer.add_uint32('tokenizer.ggml.vocab_size', vocab)
    writer.add_uint32('tokenizer.ggml.bos_token_id', 151643)
    writer.add_uint32('tokenizer.ggml.eos_token_id', 151643)

    # Tensors - load one at a time to minimize memory
    print(f"  Writing {len(tensor_names)} tensors...")
    for name in tensor_names:
        tensor = load_tensor(name)
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() == 1 and tensor.numel() < 1000:
            continue  # Skip small 1D tensors like position ids
        gguf_name = map_to_gguf_name(name)
        writer.add_tensor(gguf_name, tensor.half())
        del tensor  # Free memory

    t0 = time.time()
    writer.write()
    t1 = time.time()
    size_mb = os.path.getsize(gguf_path) / 1e6
    print(f"  Written: {size_mb:.1f} MB in {t1-t0:.1f}s")

    # Step 4: Test with llama-cli
    print("\n[4] Testing with llama-cli...")
    llama_cli = os.path.join(LLAMA_CLI_DIR, 'llama-cli.exe')

    if not os.path.exists(llama_cli):
        print(f"  llama-cli not found at {llama_cli}")
        return 1

    cmd = [llama_cli, '-m', gguf_path, '-n', '32', '-p', 'Hello world', '--log-disable']
    print(f"  Running: {' '.join(cmd[:4])}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("  SUCCESS! llama-cli loaded the model")
            print(f"  Output: {result.stdout[:200]}")
        else:
            print(f"  FAILED with code {result.returncode}")
            print(f"  Error: {result.stderr[:300]}")
            return 1
    except subprocess.TimeoutExpired:
        print("  Timed out")
        return 1
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    # Step 5: Quantize to Q4_K_M
    print("\n[5] Quantizing to Q4_K_M...")
    llama_quantize = os.path.join(LLAMA_CLI_DIR, 'llama-quantize.exe')

    q4_path = os.path.join(OUTPUT_DIR, 'Qwen2.5-0.5B-Q4_K_M.gguf')
    cmd = [llama_quantize, gguf_path, q4_path, 'Q4_K_M']

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            size_q4 = os.path.getsize(q4_path) / 1e6
            print(f"  Q4_K_M created: {size_q4:.1f} MB")
        else:
            print(f"  Quantization failed: {result.stderr[:300]}")
    except Exception as e:
        print(f"  Quantization error: {e}")

    # Step 6: Test Q4_K_M
    print("\n[6] Testing Q4_K_M with llama-cli...")
    cmd = [llama_cli, '-m', q4_path, '-n', '32', '-p', 'Hello world', '--log-disable']

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("  SUCCESS! Q4_K_M model works")
            print(f"  Output: {result.stdout[:200]}")
        else:
            print(f"  Q4_K_M FAILED: {result.stderr[:300]}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())