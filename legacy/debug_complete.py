import torch
import numpy as np

print("=" * 60)
print("DEBUG: FABQ-RC to GGUF Shape Mismatch Analysis")
print("=" * 60)

# Load FABQ-RC state dict
state = torch.load('quantized_mistral7b_fabqrc.pth', map_location='cpu', weights_only=False)
print(f"\nFABQ-RC state dict loaded: {len(state)} keys")

# Examine the first layer's structure
layer = 'model.layers.0.self_attn.q_proj'

# Check what keys exist for this layer
layer_keys = [k for k in state.keys() if k.startswith(layer + '.')]
print(f"\nKeys for {layer}:")
for k in layer_keys:
    v = state[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: type={type(v)}")

# Get component shapes
int8_channels = state[layer + '.int8_channels']
binary_channels = state[layer + '.binary_channels']
int8_weights = state[layer + '.int8_weights']
int8_scales = state[layer + '.int8_scales']
binary_recon = state[layer + '.binary_reconstructed_weights']

print(f"\n{int8_channels.shape[0]=} int8 channels")
print(f"{binary_channels.shape[0]=} binary channels")
print(f"Total out_channels = {int8_channels.shape[0] + binary_channels.shape[0]}")

print(f"\n{int8_weights.shape=} (should be n_int8 x in_channels)")
print(f"{binary_recon.shape=} (should be n_binary x in_channels)")

# Check which channels are int8 vs binary
print(f"\nFirst 10 int8 channel indices: {int8_channels[:10].tolist()}")
print(f"First 10 binary channel indices: {binary_channels[:10].tolist()}")

# Expected shape for Mistral-7B q_proj
print("\n" + "=" * 60)
print("EXPECTED SHAPES for Mistral-7B q_proj:")
print("  weight shape: (4096, 4096)")
print("=" * 60)

# Try to reconstruct properly
print("\n" + "=" * 60)
print("RECONSTRUCTING LAYER:")
print("=" * 60)

n_int8 = len(int8_channels)
n_binary = len(binary_channels)
out_channels = n_int8 + n_binary
in_channels = int8_weights.shape[1] if n_int8 > 0 else binary_recon.shape[1]

print(f"n_int8={n_int8}, n_binary={n_binary}")
print(f"out_channels={out_channels}, in_channels={in_channels}")

# Method 1: The way convert_fabqrc_complete.py does it
weight_method1 = np.zeros((out_channels, in_channels), dtype=np.float32)
int8_ch_list = int8_channels.tolist()
for i, ch in enumerate(int8_ch_list):
    scale = int8_scales[i].item()
    weight_method1[ch, :] = int8_weights[i].numpy().astype(np.float32) * scale

binary_idx = 0
for c in range(out_channels):
    if c not in set(int8_ch_list):
        weight_method1[c, :] = binary_recon[binary_idx].numpy()
        binary_idx += 1

print(f"\nMethod 1 (convert_fabqrc_complete.py):")
print(f"  Shape: {weight_method1.shape}")

# Method 2: Using tensor operations (from FABQ_RC_Paperspace.ipynb)
weight_method2 = np.zeros((out_channels, in_channels), dtype=np.float32)
if n_int8 > 0:
    # This is what Paperspace does: int8_weights.float() * int8_scales.unsqueeze(-1)
    int8_w = int8_weights.float().numpy()
    int8_s = int8_scales.unsqueeze(-1).float().numpy()  # shape (n_int8, 1)
    weight[int8_channels] = int8_w * int8_s
    
print(f"  Note: PaperSpace uses channel indexing directly")
print(f"  int8_weights shape: {int8_weights.shape}")
print(f"  int8_scales shape: {int8_scales.shape}")
print(f"  int8_scales.unsqueeze(-1) shape: {int8_scales.unsqueeze(-1).shape}")

# Check if the indices match
print(f"\nChannel indices match check:")
print(f"  int8_channels type: {type(int8_channels)}")
print(f"  binary_channels type: {type(binary_channels)}")
if isinstance(int8_channels, torch.Tensor):
    print(f"  int8_channels is Tensor, list[:5]: {int8_channels[:5].tolist()}")
else:
    print(f"  int8_channels is list, [:5]: {int8_channels[:5]}")

# The issue: binary_recon is shape (n_binary, 1) - it's storing only 1 column!
# This suggests the in_channels is 1, which is wrong
print(f"\n!!! CRITICAL ISSUE !!!")
print(f"  binary_recon.shape = {binary_recon.shape}")
print(f"  This means in_channels = 1 (WRONG!)")
print(f"  Expected: in_channels = 4096")

# Check other layers to see if they have the same issue
print("\n" + "=" * 60)
print("CHECKING OTHER LAYERS:")
print("=" * 60)

for layer_name in ['model.layers.0.self_attn.k_proj', 'model.layers.0.mlp.gate_proj']:
    if layer_name + '.int8_channels' in state:
        br = state[layer_name + '.binary_reconstructed_weights']
        iw = state[layer_name + '.int8_weights']
        print(f"{layer_name}:")
        print(f"  int8_weights: {iw.shape}")
        print(f"  binary_recon: {br.shape}")