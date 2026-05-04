import json
import sys

# Load the copy
with open('FABQ-RC-Dense-27B-Notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_model = 'Qwen/Qwen3.6-35B-A3B'
new_model = 'Qwen/Qwen3.6-27B'

changes = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if old_model in source:
            new_source = source.replace(old_model, new_model)
            cell['source'] = [new_source]
            changes += source.count(old_model)
            print(f'Changed {old_model} -> {new_model}', file=sys.stderr)

# Update notebook metadata/title
nb['metadata']['display_name'] = 'FABQ-RC: Fisher-Adaptive Binary Quantization (Dense 27B)'

print(f'Total replacements: {changes}', file=sys.stderr)

# Save
with open('FABQ-RC-Dense-27B-Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done!', file=sys.stderr)