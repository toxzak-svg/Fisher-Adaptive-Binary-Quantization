from gguf import GGUFReader
r = GGUFReader('mistral7b-fabqrc.gguf')
print('Fields count:', len(r.fields))
print('Tensors count:', len(r.tensors))
print('\nFirst 20 fields:')
for k, v in list(r.fields.items())[:20]:
    print(f'  {k}: {v}')
print('\nFirst 20 tensors:')
for i, t in enumerate(r.tensors[:20]):
    print(f'  {i}: name={t.name}, shape={t.shape}, dtype={t.tensor_type}')