import json

def final_cleanup_removes(nb_path, name):
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
        stripped = src.strip()
        
        # Drop standalone import cells that are isolated
        standalone_imports = [
            'import gc',
            'import torch',
            'import matplotlib.pyplot as plt',
            'import matplotlib.patches as mpatches'
        ]
        
        if stripped in standalone_imports:
            dropped.append(f"Dropped cell {i}: standalone {stripped}")
            continue
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return dropped, len(nb['cells'])

print("=== Final cleanup: remove ALL standalone imports ===")
d1, c1 = final_cleanup_removes('FABQ-RC-Dense-27B-Notebook.ipynb', 'Dense-27B')
for x in d1:
    print(f"  {x}")
print(f"Result: {c1} cells\n")

d2, c2 = final_cleanup_removes('Main-FABQ-RC-Notebook.ipynb', 'Main')
for x in d2:
    print(f"  {x}")
print(f"Result: {c2} cells\n")

print("Done.")