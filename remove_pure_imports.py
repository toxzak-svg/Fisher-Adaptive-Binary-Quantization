import json

def remove_pure_standalone_imports(nb_path):
    """Remove cells that are ONLY import statements with no other code."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    dropped = []
    
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            new_cells.append(cell)
            continue
        
        src = ''.join(cell.get('source', []))
        lines = src.strip().split('\n')
        
        # Check if cell is ONLY import statements (one or more)
        pure_import_lines = [l.strip() for l in lines if l.strip()]
        is_pure_imports = all(
            l.startswith('import ') or l.startswith('from ')
            for l in pure_import_lines
        )
        
        # Also check if it has NO empty lines and no other statements
        if is_pure_imports and len(pure_import_lines) >= 1:
            # Check if cell is just imports with no substantial other code
            # Only drop if it's a single standalone import line
            if len(pure_import_lines) == 1 and pure_import_lines[0] in [
                'import gc',
                'import torch',
                'import matplotlib.pyplot as plt',
                'import matplotlib.patches as mpatches'
            ]:
                dropped.append(f"Dropped cell {i}: pure import {pure_import_lines[0]}")
                continue
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return dropped

# Revert first - reload original
import shutil

# We need to start fresh from before the last cleanup since it did nothing
# Actually the last run DID drop cells (49 vs 51), so the drops did happen

# Instead let's just do targeted removal
print("=== Removing pure standalone imports ===")
d1 = remove_pure_standalone_imports('FABQ-RC-Dense-27B-Notebook.ipynb')
for x in d1:
    print(f"  {x}")
print(f"Total dropped: {len(d1)}")

d2 = remove_pure_standalone_imports('Main-FABQ-RC-Notebook.ipynb')
for x in d2:
    print(f"  {x}")
print(f"Total dropped: {len(d2)}")

print("\nDone.")