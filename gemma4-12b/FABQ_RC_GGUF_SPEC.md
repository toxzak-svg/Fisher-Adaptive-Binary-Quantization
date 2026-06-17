# FABQ-RC GGUF Integration Specification (Gemma 4 12B)

## Overview

This is a Gemma 4 12B-specific note appended to the parent project's
FABQ_RC_GGUF_SPEC.md. The FABQ-RC tensor type and codebook layout are
identical to the main spec — what changes for Gemma is the GGUF architecture
key and a couple of config adaptations for tied embeddings.

## GGUF Architecture Key

For Gemma 4 12B, the GGUF `general.architecture` is `gemma3` (Gemma 4
inherits the Gemma 3 architecture family in llama.cpp at the time of
writing). If llama.cpp renames it to `gemma4` in a later release, update
this key — the rest of the tensor layout stays the same.

```python
writer.add_key_string("general.architecture", "gemma3")
writer.add_key_string("general.name", "Gemma-4-12B-FABQ-RC")
```

## Tied Embeddings

Gemma 4 ties the input embedding and the output projection (`lm_head`).
Practically:

- `model.embed_tokens.weight` and `lm_head.weight` are the **same tensor**
  in the source checkpoint.
- FABQ-RC skips them in the linear sweep (kept in original precision).
- When exporting to GGUF, emit a single `token_embd.weight` tensor and **do
  not** emit a separate `output.weight` — set `output.weight` as a metadata
  pointer to `token_embd.weight` per the llama.cpp Gemma convention.

This avoids emitting the embedding twice (which would inflate the file).

## GGUF Metadata for Gemma 4 12B

```python
cfg = model.config.to_dict()
hpp     = cfg["hidden_size"]
nhead   = cfg["num_attention_heads"]
nkv     = cfg["num_key_value_heads"]
nlayer  = cfg["num_hidden_layers"]
inter   = cfg["intermediate_size"]
vocab   = cfg["vocab_size"]
rope    = cfg.get("rope_theta", 10000.0)
head_dim = hpp // nhead

writer.add_key_uint32("gemma3.embedding_length",   hpp)
writer.add_key_uint32("gemma3.hidden_length",      hpp)
writer.add_key_uint32("gemma3.intermediate_length", inter)
writer.add_key_uint32("gemma3.num_attention_heads", nhead)
writer.add_key_uint32("gemma3.num_key_value_heads", nkv)
writer.add_key_uint32("gemma3.num_hidden_layers",   nlayer)
writer.add_key_uint32("gemma3.context_length",      4096)
writer.add_key_uint32("gemma3.vocabulary_size",     vocab)
writer.add_key_float32("gemma3.rope_theta",         rope)
writer.add_key_uint32("gemma3.attention.head_dim",  head_dim)
```

## Tensor naming

Gemma 4 12B layer naming convention (matching HuggingFace `Gemma3ForCausalLM`):

```
model.embed_tokens.weight         # shared with lm_head
model.layers.{N}.input_layernorm.weight
model.layers.{N}.self_attn.q_proj.weight
model.layers.{N}.self_attn.k_proj.weight
model.layers.{N}.self_attn.v_proj.weight
model.layers.{N}.self_attn.o_proj.weight
model.layers.{N}.self_attn.q_norm.weight
model.layers.{N}.self_attn.k_norm.weight
model.layers.{N}.post_attention_layernorm.weight
model.layers.{N}.mlp.gate_proj.weight
model.layers.{N}.mlp.up_proj.weight
model.layers.{N}.mlp.down_proj.weight
model.layers.{N}.pre_feedforward_layernorm.weight
model.layers.{N}.post_feedforward_layernorm.weight
model.norm.weight
```

For GGUF export, the layer prefix is `blk.{N}.` instead of
`model.layers.{N}.`:

```
token_embd.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
blk.0.attn_q_norm.weight
blk.0.attn_k_norm.weight
blk.0.attn_norm.weight
blk.0.ffn_gate.weight
blk.0.ffn_up.weight
blk.0.ffn_down.weight
blk.0.ffn_norm.weight
...
output_norm.weight
```

## FABQ-RC Tensors

The same custom tensor type as the main spec (GGML_TYPE_FABQ_RC = 41):

- `fabq.codebook.0` … `fabq.codebook.3` — 4 tiered codebooks
  (Fisher quartile-based), float32[64, blocksize] each
- `fabq.layer.{N}.blocksize` — u32
- `fabq.layer.{N}.int4_channels` — u32
- `fabq.layer.{N}.binary_channels` — u32
- `fabq.layer.{N}.fisher_quartile` — u32 (0-3)

## Dequantization

Identical to the parent spec. The `dequantize_fabq_rc` function in
`ggml-quants.c` is model-agnostic — it only cares about the block layout.

## File size estimate

For Gemma 4 12B (~12.5B params, 36 layers, hidden 3072):

- ~12.5B params × 1.21 bpw = ~15.1 Gbit ≈ **1.9 GB** quantized
- vs FP16: 25 GB
- compression ratio: ~13.2×

## Reference

- Parent spec: `../FABQ_RC_GGUF_SPEC.md`
- llama.cpp Gemma 3 implementation: `llama.cpp/models/llama.cpp:gemma3`
  (gemma4 support expected to land in the same file)
