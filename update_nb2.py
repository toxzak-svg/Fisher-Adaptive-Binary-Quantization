#!/usr/bin/env python3
"""Update Main-FABQ-RC-Notebook.ipynb with corrected architecture."""
import json

# Read notebook
with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = []

# 1. Update INT8_FRACTION to INT4_FRACTION in code cells
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    for i, line in enumerate(cell['source']):
        if 'INT8_FRACTION' in line:
            cell['source'][i] = line.replace('INT8_FRACTION', 'INT4_FRACTION')
            changes.append(f'Code cell INT8_FRACTION -> INT4_FRACTION at line {i}')
        if "int8_fraction" in line:
            cell['source'][i] = line.replace('int8_fraction', 'int4_fraction')
            changes.append(f'Code cell int8_fraction -> int4_fraction at line {i}')

# 2. Update codebook size references (256 -> 4x64)
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    for i, line in enumerate(cell['source']):
        if 'n_clusters=256' in line:
            cell['source'][i] = line.replace('n_clusters=256', 'n_clusters=64')
            changes.append(f'Code cell n_clusters=256 -> 64 at line {i}')
        if '= 256' in line and 'cluster' in line.lower():
            cell['source'][i] = line.replace('= 256', '= 64')
            changes.append(f'Code cell cluster 256 -> 64 at line {i}')

# 3. Update markdown cells
for cell in nb['cells']:
    if cell['cell_type'] != 'markdown':
        continue
    for i, line in enumerate(cell['source']):
        # Update bpw target
        if '~1.15' in line and 'bpw' in line:
            cell['source'][i] = line.replace('~1.15–1.20', '~1.21')
            changes.append(f'Markdown updated bpw in line {i}')
        # Update codebook description
        if '256 centroids, shared across layers' in line:
            cell['source'][i] = line.replace('256 centroids, shared across layers', '4 tiered codebooks × 64 centroids each')
            changes.append(f'Markdown updated codebook description in line {i}')
        # Update int8 to int4 in stage 2 description
        if 'Top 5% channels → int8' in line:
            cell['source'][i] = line.replace('int8', 'int4')
            changes.append(f'Markdown updated int8 -> int4 in line {i}')

# Save
with open('Main-FABQ-RC-Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False)

print(f'Made {len(changes)} changes:')
for c in changes:
    print(f'  {c}')
print('Done!')