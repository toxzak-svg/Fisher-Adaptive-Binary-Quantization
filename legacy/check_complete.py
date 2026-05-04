from gguf import GGUFReader
r = GGUFReader('mistral7b-fabqrc-complete.gguf')
print('Fields:', len(r.fields))
print('Tensors:', len(r.tensors))
print('Key fields:')
for k in ['general.architecture', 'mistral.vocab_size', 'tokenizer.ggml.model']:
    if k in r.fields:
        print(f'  {k}: present')
    else:
        print(f'  {k}: MISSING')