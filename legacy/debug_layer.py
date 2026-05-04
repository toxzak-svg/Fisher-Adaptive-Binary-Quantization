import torch

s = torch.load('quantized_mistral7b_fabqrc.pth', map_location='cpu', weights_only=False)

layer = 'model.layers.0.self_attn.q_proj'
print(f'Layer: {layer}')
print(f'  int8_channels: {s[layer + ".int8_channels"].shape}')
print(f'  int8_weights: {s[layer + ".int8_weights"].shape}')
print(f'  int8_scales: {s[layer + ".int8_scales"].shape}')
print(f'  binary_channels: {s[layer + ".binary_channels"].shape}')
print(f'  binary_reconstructed_weights: {s[layer + ".binary_reconstructed_weights"].shape}')

n_int8 = len(s[layer + '.int8_channels'])
n_binary = s[layer + '.binary_reconstructed_weights'].shape[0]
print(f'  Total output channels: {n_int8 + n_binary}')
in_ch = s[layer + ".int8_weights"].shape[1] if n_int8 > 0 else s[layer + ".binary_reconstructed_weights"].shape[1]
print(f'  Input channels: {in_ch}')