"""
Final cleanup of FABQ-RC notebooks - remove all standalone duplicate import cells.
"""

import json

def final_cleanup(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    dropped = []
    
    seen_imports = {}  # import_line -> first_cell_index
    
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            new_cells.append(cell)
            continue
        
        src = ''.join(cell.get('source', []))
        stripped = src.strip()
        
        # Check for standalone single import lines
        single_imports = [
            'import gc',
            'import torch',
            'import matplotlib.pyplot as plt',
        ]
        
        is_standalone_import = any(stripped == imp for imp in single_imports)
        
        if is_standalone_import:
            # Drop if we've seen this import before
            if stripped in seen_imports:
                dropped.append(f"Dropped duplicate '{stripped}' (cell {i}, first was {seen_imports[stripped]})")
                continue
            seen_imports[stripped] = i
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return dropped

print("=== Final Cleanup FABQ-RC-Dense-27B-Notebook.ipynb ===")
d1 = final_cleanup('FABQ-RC-Dense-27B-Notebook.ipynb')
for d in d1:
    print(f"  {d}")
print(f"Total dropped: {len(d1)}")

print()
print("=== Final Cleanup Main-FABQ-RC-Notebook.ipynb ===")
d2 = final_cleanup('Main-FABQ-RC-Notebook.ipynb')
for d in d2:
    print(f"  {d}")
print(f"Total dropped: {len(d2)}")

print("\nDone.")