"""
Clean and debug both FABQ-RC notebooks.
Issues identified:
1. FABQ-RC-Dense-27B-Notebook.ipynb cell 19: BS_CANDIDATES has old [16,32,64,128,256], should be [64,128,256,512]
2. FABQ-RC-Dense-27B-Notebook.ipynb cell 21: INT8_FRACTION should be INT4_FRACTION
3. FABQ-RC-Dense-27B-Notebook.ipynb cell 6: "Swapped to Qwen3.6-35B-A3B" - wrong model for 27B notebook
4. FABQ-RC-Dense-27B-Notebook.ipynb cell 7: "### 3.1b Scaling to 35B Model" - wrong for 27B
5. Both notebooks have many duplicate/cleanup `import gc` cells
6. Main notebook cell 22: INT4_FRACTION comment still says "int8"
7. Main notebook cell 32-33: duplicate HfApi import cells
"""

import json
import re

def clean_notebook(nb_path, is_27b=False):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    changes = []
    
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        
        # Fix 1: BS_CANDIDATES in FABQ-RC-Dense-27B-Notebook
        if 'BS_CANDIDATES = [16, 32, 64, 128, 256]' in src:
            new_src = src.replace('BS_CANDIDATES = [16, 32, 64, 128, 256]', 'BS_CANDIDATES = [64, 128, 256, 512]')
            cell['source'] = [new_src]
            changes.append(f"Cell {i}: Fixed BS_CANDIDATES -> [64, 128, 256, 512]")
        
        # Fix 2: INT8_FRACTION -> INT4_FRACTION (Dense 27B)
        if 'INT8_FRACTION = 0.05' in src:
            new_src = src.replace('INT8_FRACTION = 0.05', 'INT4_FRACTION = 0.05')
            cell['source'] = [new_src]
            changes.append(f"Cell {i}: INT8_FRACTION -> INT4_FRACTION")
        
        # Fix 3: INT4_FRACTION comment still says int8
        if 'INT4_FRACTION = 0.05' in src and 'int8' in src:
            new_src = src.replace('# Keep top 5% Fisher channels as int8', '# Keep top 5% Fisher channels as int4')
            cell['source'] = [new_src]
            changes.append(f"Cell {i}: Fixed int8 comment -> int4")
        
        # Fix 4: 35B model references in 27B notebook
        if is_27b and ('Qwen3.6-35B-A3B' in src or '35B' in src):
            if 'Swapped to Qwen3.6-35B-A3B' in src:
                new_src = src.replace('Swapped to Qwen3.6-35B-A3B for testing', 'Quantizing Qwen3.6-27B')
                cell['source'] = [new_src]
                changes.append(f"Cell {i}: Fixed model reference 35B -> 27B")
            elif '### 3.1b Scaling to 35B Model' in src:
                new_src = src.replace('### 3.1b Scaling to 35B Model', '### 3.1b Scaling to 27B Model')
                cell['source'] = [new_src]
                changes.append(f"Cell {i}: Fixed section header 35B -> 27B")
    
    # Remove duplicate consecutive `import gc` cells (keep only first in sequence)
    new_cells = []
    prev_was_gc = False
    for cell in cells:
        src = ''.join(cell.get('source', []))
        is_gc_only = src.strip() == 'import gc' or src.strip() == 'import gc\n'
        
        if is_gc_only and prev_was_gc:
            changes.append(f"Dropped duplicate import gc cell")
            continue
        new_cells.append(cell)
        prev_was_gc = is_gc_only
    
    # Remove duplicate HfApi import cells
    hc = 0
    final_cells = []
    for cell in new_cells:
        src = ''.join(cell.get('source', []))
        if 'from huggingface_hub import HfApi, create_repo' in src:
            hc += 1
            if hc > 1:
                changes.append("Dropped duplicate HfApi cell")
                continue
        final_cells.append(cell)
    
    nb['cells'] = final_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return changes

print("=== Cleaning FABQ-RC-Dense-27B-Notebook.ipynb ===")
changes_27b = clean_notebook('FABQ-RC-Dense-27B-Notebook.ipynb', is_27b=True)
for c in changes_27b:
    print(f"  {c}")
print(f"Total changes: {len(changes_27b)}")

print()
print("=== Cleaning Main-FABQ-RC-Notebook.ipynb ===")
changes_main = clean_notebook('Main-FABQ-RC-Notebook.ipynb', is_27b=False)
for c in changes_main:
    print(f"  {c}")
print(f"Total changes: {len(changes_main)}")

print("\nDone cleaning notebooks.")