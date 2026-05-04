import torch

print("=" * 60)
print("ROOT CAUSE ANALYSIS: FABQ-RC Shape Bug")
print("=" * 60)

# Load FABQ-RC state dict
state = torch.load('quantized_mistral7b_fabqrc.pth', map_location='cpu', weights_only=False)

# The issue: int8_weights and binary_reconstructed_weights have wrong shape
# They should be (n_channels, 4096) but are (n_channels, 1)

layer = 'model.layers.0.self_attn.q_proj'

print(f"\nLayer: {layer}")
print(f"  int8_weights.shape = {state[layer + '.int8_weights'].shape}")
print(f"  binary_reconstructed_weights.shape = {state[layer + '.binary_reconstructed_weights'].shape}")
print(f"  Expected: (n_channels, 4096)")

# This explains the GGUF output:
# GGUF has shape [1, 4096] because we reconstruct to (4096, 1) then transpose?

# Let's check what the original model weights look like
# The conversion script does:
#   weight_fp16 = reconstruct_layer_fp16(state_dict, layer_name)
#   writer.add_tensor(tensor_name, weight_fp16, raw_shape=weight_fp16.shape)
#
# And reconstruct_layer_fp16 creates:
#   weight = np.zeros((out_channels, in_channels), dtype=np.float32)
#
# With out_channels = 4096 and in_channels = 1 (from int8_weights.shape[1])
# So we get shape (4096, 1)

print("\n" + "=" * 60)
print("THE BUG: Quantization stored weights with in_channels=1")
print("=" * 60)
print("""
This happened during FABQ-RC quantization - the weight matrices were
incorrectly processed, resulting in binary_reconstructed_weights having
shape (3892, 1) instead of (3892, 4096).

Possible causes:
1. Transpose operation bug during quantization
2. Incorrect slicing when extracting weight blocks
3. Wrong dimension used when processing blocks

The GGUF file is correct in structure (225 tensors) but each weight
tensor has the wrong shape (1, 4096) instead of (4096, 4096).
""")

# Verify the tensor count is correct
print("=" * 60) 
print("GGUF TENSOR COUNT VERIFICATION:")
print("=" * 60)

# Mistral-7B expected structure:
# - token_embd.weight: 1 tensor
# - 32 layers x 7 tensors = 224 tensors
#   (attn_q, attn_k, attn_v, attn_output, mlp_gate, mlp_up, mlp_down)
# - output.weight: 1 tensor
# Total: 226 tensors

# But we have 225, which means one might be missing

print("""
Expected tensors:
  - token_embd.weight: 1
  - blk.N.attn_q.weight: 32  
  - blk.N.attn_k.weight: 32
  - blk.N.attn_v.weight: 32
  - blk.N.attn_output.weight: 32
  - blk.N.mlp.gate.weight: 32
  - blk.N.mlp.up.weight: 32
  - blk.N.mlp.down.weight: 32
  - output.weight: 1
Total: 226

Actual GGUF has: 225
""")

# Check if there's an issue with the last layer
print("Checking if all 32 layers are present...")
for i in range(32):
    tensor_name = f'blk.{i}.attn_q.weight'
    found = any(t.name == tensor_name for t in __import__('gguf').GGUFReader('mistral7b-fabqrc-complete.gguf').tensors)
    if not found:
        print(f"  MISSING: {tensor_name}")