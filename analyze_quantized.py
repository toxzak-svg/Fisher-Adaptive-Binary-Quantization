import torch
d=torch.load('quantized_mistral7b_fabqrc.pth',map_location='cpu',weights_only=False)

# Compute bpw properly
total_bits=0
total_params=0
n_int8_total=0
n_binary_total=0

for k in d.keys():
    if '.int8_channels' in k:
        ln=k.replace('.int8_channels','')
        n_int8=len(d[k])
        n_binary=len(d[ln+'.binary_channels'])
        n_int8_total+=n_int8
        n_binary_total+=n_binary
        if n_int8>0:
            in_feat=d[ln+'.int8_weights'].shape[1]
        else:
            in_feat=d[ln+'.binary_reconstructed_weights'].shape[1]
        out_feat=n_int8+n_binary
        n=out_feat*in_feat
        total_params+=n
        total_bits+=n_int8*in_feat*8 + n_binary*in_feat*1

bpw=total_bits/total_params
print(f'Total int8 channels: {n_int8_total}')
print(f'Total binary channels: {n_binary_total}')
print(f'Total params: {total_params:,}')
print(f'Bpw: {bpw:.4f}')