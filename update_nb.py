#!/usr/bin/env python3
"""Update Main-FABQ-RC-Notebook.ipynb with corrected architecture."""
import json, re

# Read notebook
with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = 0

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    
    # 1. Update BS_CANDIDATES
    if 'BS_CANDIDATES = [16, 32, 64, 128, 256]' in source:
        for i, line in enumerate(cell['source']):
            if 'BS_CANDIDATES = [16, 32, 64, 128, 256]' in line:
                cell['source'][i] = line.replace('[16, 32, 64, 128, 256]', '[64, 128, 256, 512]')
                changes += 1
    
    # 2. Update BS_PENALTIES
    if 'BS_PENALTIES = {16: 1.5, 32: 1.2, 64: 1.1, 128: 1.0, 256: 0.9}' in source:
        for i, line in enumerate(cell['source']):
            if 'BS_PENALTIES = {16: 1.5, 32: 1.2, 64: 1.1, 128: 1.0, 256: 0.9}' in line:
                cell['source'][i] = line.replace('{16: 1.5, 32: 1.2, 64: 1.1, 128: 1.0, 256: 0.9}', '{64: 1.1, 128: 1.0, 256: 0.9, 512: 0.85}')
                changes += 1

print(f'Made {changes} code cell changes')

# Save
with open('Main-FABQ-RC-Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False)

# Verify
with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

has_new = 'BS_CANDIDATES = [64, 128, 256, 512]' in content
has_old = 'BS_CANDIDATES = [16, 32, 64, 128, 256]' in content
print(f'Has new BS_CANDIDATES: {has_new}')
print(f'Has old BS_CANDIDATES: {has_old}')

# Also update markdown cells that reference the old blocksize candidates
with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

md_changes = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'markdown':
        continue
    for i, line in enumerate(cell['source']):
        if '{16, 32, 64, 128, 256}' in line:
            cell['source'][i] = line.replace('{16, 32, 64, 128, 256}', '{64, 128, 256, 512}')
            md_changes += 1
        if '~1.15' in line and 'bpw' in line:
            cell['source'][i] = line.replace('~1.15–1.20', '~1.21')
            md_changes += 1

print(f'Made {md_changes} markdown changes')

with open('Main-FABQ-RC-Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False)

print('Done!')