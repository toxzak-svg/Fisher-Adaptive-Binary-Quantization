from gguf import GGUFReader

r = GGUFReader('mistral7b-fabqrc-complete.gguf')
print(f'Total tensors: {len(r.tensors)}')

# Find blk.0.attn_q.weight tensor
for t in r.tensors:
    if 'blk.0.attn_q.weight' in t.name:
        print(f'Tensor: {t.name}')
        print(f'  Shape: {t.shape}')
        print(f'  Tensor type: {t.tensor_type}')
        break

# Print first 30 tensors with their shapes
print('\nFirst 30 tensors:')
for i, t in enumerate(r.tensors[:30]):
    print(f'  {i}: {t.name} shape={t.shape} type={t.tensor_type}')