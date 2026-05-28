import json

def remove_reload_cycle(nb_path):
    """
    The reload cycle cells (33-38) are dead code - after quantization you should 
    just save and exit, not reload from HF. Remove all of them.
    """
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    dropped = []
    
    for i, cell in enumerate(cells):
        # Remove cells 33-38 (the entire reload/re-init/validate cycle)
        if i >= 33 and i <= 38:
            dropped.append(f"Dropped cell {i}: reload cycle dead code")
            continue
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return dropped, len(nb['cells'])

print("=== Removing reload cycle (cells 33-38) ===")
d1, c1 = remove_reload_cycle('FABQ-RC-Dense-27B-Notebook.ipynb')
for x in d1:
    print(f"  {x}")
print(f"Result: {c1} cells")

print()
d2, c2 = remove_reload_cycle('Main-FABQ-RC-Notebook.ipynb')
for x in d2:
    print(f"  {x}")
print(f"Result: {c2} cells")

print("\nDone.")