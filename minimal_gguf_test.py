#!/usr/bin/env python3
"""
Minimal GGUF test - creates a tiny GGUF to verify llama-cli works.
Then tests quantize.
"""
import os
import struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
ALIGN = 32

def align_to(x, a):
    return ((x + a - 1) // a) * a

def write_str(f, s):
    enc = s.encode('utf-8')
    f.write(struct.pack('<Q', len(enc)))
    f.write(enc)

# Create tiny test GGUF
path = r"C:\Users\Zwmar\AppData\Local\Temp\fabq-test\test.gguf"

with open(path, 'wb') as f:
    # Header
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))
    f.write(struct.pack('<Q', 1))  # 1 tensor
    f.write(struct.pack('<Q', 12))  # 12 KV pairs

    # Metadata
    write_str(f, 'general.architecture')
    f.write(struct.pack('<I', GGUF_TYPE_STRING)); write_str(f, 'qwen2')

    write_str(f, 'general.name')
    f.write(struct.pack('<I', GGUF_TYPE_STRING)); write_str(f, 'test')

    write_str(f, 'qwen2.context_length')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 512))

    write_str(f, 'qwen2.vocab_size')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 1000))

    write_str(f, 'qwen2.embedding_length')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 128))

    write_str(f, 'qwen2.block_count')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 2))

    write_str(f, 'qwen2.feed_forward_length')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 512))

    write_str(f, 'qwen2.attention.head_count')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 4))

    write_str(f, 'qwen2.attention.head_count_kv')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 2))

    write_str(f, 'qwen2.rope.dimension_count')
    f.write(struct.pack('<I', GGUF_TYPE_UINT32)); f.write(struct.pack('<I', 32))

    write_str(f, 'qwen2.rope.freq_base')
    f.write(struct.pack('<I', GGUF_TYPE_FLOAT32)); f.write(struct.pack('<f', 10000.0))

    write_str(f, 'qwen2.attention.layer_norm_rms_epsilon')
    f.write(struct.pack('<I', GGUF_TYPE_FLOAT32)); f.write(struct.pack('<f', 1e-5))

    write_str(f, 'tokenizer.ggml.model')
    f.write(struct.pack('<I', GGUF_TYPE_STRING)); write_str(f, 'llama')

    # Tensor info - just a small FP16 tensor
    write_str(f, 'token_embd.weight')
    f.write(struct.pack('<I', 2))  # 2D
    f.write(struct.pack('<Q', 128))  # dim0
    f.write(struct.pack('<Q', 1000))  # dim1
    f.write(struct.pack('<I', 1))  # type F16
    f.write(struct.pack('<Q', 0))  # offset

    # Padding
    f.write(b'\x00' * (align_to(f.tell(), ALIGN) - f.tell()))

    # Tensor data - 128 * 1000 * 2 = 256000 bytes
    f.write(b'\x00' * 256000)

print(f"Created test GGUF at {path}")
print(f"Size: {os.path.getsize(path)} bytes")