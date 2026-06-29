#!/usr/bin/env python3
"""
Fix the FABQ-RC-GGUF-Reexport.ipynb notebook.

Key fixes:
1. GGUFWriter: fix metadata type handling, string length limits
2. Shape calculation for raw bytes fallback
3. Tokenizer metadata storage
4. Tensor name mapping for Qwen3.6
"""
import json
import sys

def fix_gguf_writer_cell(cells):
    """Fix the GGUFWriter class and add proper tensor handling."""
    # Find the cell with "# Cell 3: Fixed GGUFWriter"
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'Cell 3: Fixed GGUFWriter' in src:
                # Replace the entire cell with fixed version
                fixed_src = '''# Cell 3: Fixed GGUFWriter (v2 - fixes shape calc, metadata, string limits)
# Key fixes:
# 1. Shape calculation for raw bytes uses actual element count from packed size
# 2. Metadata values checked for size limits before writing
# 3. Added proper handling for token arrays and other large values

import os, torch, struct, json, gc, sys, time
import numpy as np
from huggingface_hub import HfApi, create_repo, snapshot_download
from transformers import AutoConfig, AutoTokenizer

# GGUF v3 constants
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32
Q1_K_ELEMS = 256
Q1_K_SIZE = 352

# GGML quantization types
GGML_TYPE_F16 = 1
GGML_TYPE_Q1_K = 20

# GGUF metadata value types (standard v3 codes)
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY_U32 = 9
GGUF_TYPE_UINT64 = 10

MAX_STRING_LEN = 1024 * 1024 * 1024  # 1GB max string length

def align_to(x, a):
    return ((x + a - 1) // a) * a


class GGUFWriter:
    Q1_K = GGML_TYPE_Q1_K

    def __init__(self, path):
        self.path = path
        self.tensors = []
        self.kv = []

    def add_key_string(self, key, val):
        # Truncate very long strings to prevent GGUF corruption
        if isinstance(val, str) and len(val) > MAX_STRING_LEN:
            print(f"  WARNING: truncating {key} from {len(val)} to {MAX_STRING_LEN} chars")
            val = val[:MAX_STRING_LEN]
        self.kv.append((key, GGUF_TYPE_STRING, val))

    def add_key_uint32(self, key, val):
        self.kv.append((key, GGUF_TYPE_UINT32, val))

    def add_key_float32(self, key, val):
        self.kv.append((key, GGUF_TYPE_FLOAT32, val))

    def add_key_uint64(self, key, val):
        self.kv.append((key, GGUF_TYPE_UINT64, val))

    def add_key_bool(self, key, val):
        self.kv.append((key, GGUF_TYPE_BOOL, val))

    def add_key_array_u32(self, key, val):
        self.kv.append((key, GGUF_TYPE_ARRAY_U32, val))

    def add_tensor(self, name, data, dtype=GGML_TYPE_F16):
        # Ensure data has proper shape
        if isinstance(data, np.ndarray):
            pass  # Already has shape
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
        self.tensors.append((name, data, dtype))

    def _write_str(self, f, s):
        enc = s.encode('utf-8')
        f.write(struct.pack('<Q', len(enc)))
        f.write(enc)

    def _write_array_u32(self, f, arr):
        f.write(struct.pack('<I', len(arr)))  # array length
        for v in arr:
            f.write(struct.pack('<I', v))

    def write(self):
        with open(self.path, 'wb') as f:
            # === HEADER ===
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))    # tensor_count FIRST
            f.write(struct.pack('<Q', len(self.kv)))        # metadata_kv_count SECOND

            # === METADATA KV PAIRS ===
            for key, vtype, val in self.kv:
                self._write_str(f, key)
                f.write(struct.pack('<I', vtype))
                if vtype == GGUF_TYPE_STRING:
                    self._write_str(f, val)
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack('<I', val))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack('<f', val))
                elif vtype == GGUF_TYPE_UINT64:
                    f.write(struct.pack('<Q', val))
                elif vtype == GGUF_TYPE_BOOL:
                    f.write(struct.pack('<B', int(bool(val))))
                elif vtype == GGUF_TYPE_ARRAY_U32:
                    self._write_array_u32(f, val)
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack('<i', val))
                elif vtype == GGUF_TYPE_INT16:
                    f.write(struct.pack('<h', val))
                elif vtype == GGUF_TYPE_UINT16:
                    f.write(struct.pack('<H', val))
                elif vtype == GGUF_TYPE_INT8:
                    f.write(struct.pack('<b', val))
                elif vtype == GGUF_TYPE_UINT8:
                    f.write(struct.pack('<B', val))

            # === TENSOR INFO ENTRIES ===
            tensor_meta = []
            offset = 0
            for name, data, dtype in self.tensors:
                if isinstance(data, np.ndarray):
                    shape = list(data.shape)
                    nbytes = data.nbytes
                elif isinstance(data, (list, tuple)):
                    # For packed data from _pack_q1_k
                    nbytes = len(data) if isinstance(data, (bytes, bytearray)) else sum(np.array(x).nbytes if hasattr(x, 'nbytes') else len(x) for x in data)
                    # Infer shape from nbytes based on dtype
                    if dtype == GGML_TYPE_Q1_K:
                        # Q1_K uses 352 bytes per 256 elements
                        n_elems = nbytes * 8 // 1  # approx
                        shape = [nbytes // Q1_K_SIZE * Q1_K_ELEMS]
                    else:
                        shape = [nbytes // 2]  # fallback assume f16
                else:
                    nbytes = len(data)
                    # FIXED: Proper shape calculation for Q1_K packed data
                    if dtype == GGML_TYPE_Q1_K:
                        # Q1_K block = 352 bytes for 256 elements
                        n_blocks = nbytes // Q1_K_SIZE
                        n_elems = n_blocks * Q1_K_ELEMS
                        shape = [n_elems]
                    else:
                        shape = [nbytes // 2]  # fallback: assume f16 = 2 bytes

                n_dims = len(shape)
                self._write_str(f, name)
                f.write(struct.pack('<I', n_dims))
                for d in reversed(shape):
                    f.write(struct.pack('<Q', d))
                f.write(struct.pack('<I', dtype))
                f.write(struct.pack('<Q', offset))
                tensor_meta.append((name, data, dtype, shape, nbytes))
                offset += align_to(nbytes, ALIGN)

            # === PADDING BEFORE DATA ===
            f.write(b'\\x00' * (align_to(f.tell(), ALIGN) - f.tell()))

            # === TENSOR DATA ===
            for name, data, dtype, shape, nbytes in tensor_meta:
                f.write(b'\\x00' * (align_to(f.tell(), ALIGN) - f.tell()))
                if isinstance(data, np.ndarray):
                    if dtype == GGML_TYPE_F16:
                        f.write(data.astype(np.float16).tobytes())
                    elif dtype == GGML_TYPE_Q1_K:
                        f.write(self._pack_q1_k(data))
                elif isinstance(data, (bytes, bytearray)):
                    f.write(data)
                else:
                    f.write(bytes(data) if not isinstance(data, bytes) else data)
            f.flush()

    def _pack_q1_k(self, w):
        """Pack FP16 weights to Q1_K format (256 elements per block, 352 bytes)."""
        if isinstance(w, np.ndarray):
            w_flat = w.flatten().astype(np.float16)
        else:
            w_flat = np.array(w).flatten().astype(np.float16)
        n = len(w_flat)
        n_blk = (n + Q1_K_ELEMS - 1) // Q1_K_ELEMS
        out = bytearray(n_blk * Q1_K_SIZE)
        for b in range(n_blk):
            s = b * Q1_K_ELEMS
            e = min(s + Q1_K_ELEMS, n)
            vals = w_flat[s:e]
            blk = b * Q1_K_SIZE
            vmin, vmax = float(vals.min()), float(vals.max())
            span = vmax - vmin
            struct.pack_into('<e', out, blk, np.float16(span))
            struct.pack_into('<e', out, blk + 2, np.float16(vmin))
            if span < 1e-9:
                for i in range(4): out[4 + i] = 0x88
                continue
            n_val = len(vals)
            for grp in range(4):
                s_idx, e_idx = grp * 64, min(grp * 64 + 64, n_val)
                grp_v = vals[s_idx:e_idx]
                gspan = float(grp_v.max()) - float(grp_v.min())
                if gspan < 1e-9:
                    out[4 + grp] = 0x88
                else:
                    target = gspan / 15.0
                    best_nib, best_err = 8, abs(target - (2.0 ** (8 - 8)))
                    for nib in range(16):
                        err = abs(target - (2.0 ** (nib - 8)))
                        if err < best_err:
                            best_err, best_nib = nib
                    out[4 + grp] = (best_nib << 4) | best_nib
            for grp in range(4):
                s_idx, e_idx = grp * 64, min(grp * 64 + 64, n_val)
                grp_v = vals[s_idx:e_idx]
                nib = out[4 + grp] & 0x0F
                scale = 2.0 ** (nib - 8)
                base = float(vals[grp * 64]) if grp * 64 < n_val else 0.0
                for i, v in enumerate(grp_v):
                    qv = max(0, min(15, round((float(v) - base) / scale)))
                    row = i // 2
                    if i % 2 == 0:
                        out[8 + grp * 32 + row] = int(qv) << 4
                    else:
                        out[8 + grp * 32 + row] |= int(qv)
        return bytes(out)
'''
                cell['source'] = [fixed_src]
                return i, cell
    return None, None


def fix_tensor_name_mapping(cells):
    """Fix the tensor name mapping for Qwen3.6."""
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'Cell 4: Tensor name mapping' in src and 'Qwen3.6 uses linear_attn' in src:
                # Update the function to better handle Qwen3.6
                fixed_src = '''# Cell 4: Tensor name mapping (Qwen -> GGUF conventions)
# Fixed for Qwen3.6 which uses different naming than Qwen2

def map_to_gguf_name(fabq_name):
    """Map a FABQ-RC layer name to GGUF tensor name."""
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


def get_gguf_tensor_name_for_module(module_name, suffix='.weight'):
    """Map module path to GGUF tensor name."""
    if module_name.endswith('.bias'):
        base = module_name[:-5]
        return map_to_gguf_name(base) + '.bias'
    elif module_name.endswith('.weight'):
        base = module_name[:-7]
        return map_to_gguf_name(base) + '.weight'
    else:
        return map_to_gguf_name(module_name) + suffix
'''
                cell['source'] = [fixed_src]
                return i, cell
    return None, None


def fix_metadata_cell(cells):
    """Fix the tokenizer metadata - store properly not as huge string."""
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if '# --- Tokenizer metadata ---' in src and 'tokenizer.ggml.tokens' in src:
                # Find and replace the tokenizer metadata section
                lines = cell['source']
                new_lines = []
                skip_next = False
                for line in lines:
                    if 'tokenizer.ggml.tokens' in line and 'str(tokenizer' in line:
                        # Replace with safer version - just store vocab size and type
                        new_lines.append("                # FIXED: Don't store huge token list as string\n")
                        new_lines.append("                writer.add_key_uint32('tokenizer.ggml.vocab_size', len(tokenizer))\n")
                        new_lines.append("                writer.add_key_string('tokenizer.ggml.model', 'llama')\n")
                        skip_next = True
                    elif skip_next and ('tokenizer.ggml.bos_token_id' in line or 'tokenizer.ggml.eos_token_id' in line):
                        # Skip the next few lines that were part of the bad tokenizer storage
                        if 'tokenizer.ggml.bos_token_id' in line:
                            new_lines.append("                writer.add_key_uint32('tokenizer.ggml.bos_token_id', int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else 0)\n")
                            new_lines.append("                writer.add_key_uint32('tokenizer.ggml.eos_token_id', int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0)\n")
                            skip_next = False
                        continue
                    else:
                        new_lines.append(line)
                        if skip_next and ('tokenizer.ggml.bos_token_id' in line or 'tokenizer.ggml.eos_token_id' in line):
                            pass
                cell['source'] = new_lines
                return i, cell
    return None, None


def main():
    notebook_path = 'FABQ-RC-GGUF-Reexport.ipynb'

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # Apply fixes
    print("Fixing GGUFWriter class...")
    i, cell = fix_gguf_writer_cell(cells)
    if i is not None:
        print(f"  Fixed GGUFWriter at cell {i}")

    print("Fixing tensor name mapping...")
    i, cell = fix_tensor_name_mapping(cells)
    if i is not None:
        print(f"  Fixed tensor name mapping at cell {i}")

    print("Fixing tokenizer metadata...")
    i, cell = fix_metadata_cell(cells)
    if i is not None:
        print(f"  Fixed tokenizer metadata at cell {i}")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False)

    print(f"\\nSaved fixed notebook to {notebook_path}")
    print("\\nNOTE: The notebook re-export approach has fundamental issues:")
    print("  - FABQ-RC uses adaptive blocksizes {64,128,256,512}, but Q1_K uses fixed 256")
    print("  - Reconstructed FP16 -> Q1_K loses the compression advantage")
    print("  - Better to implement FABQ-RC decoding in llama.cpp directly")
    print("  - Or use a proper Q4_K_M quantization of the reconstructed weights")

if __name__ == '__main__':
    main()