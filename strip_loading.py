import json

def strip_loading_section(nb_path):
    """
    Remove the entire Loading section (cells 28-32 in Dense, 28-34 in Main).
    After quantization, you just save and you're done.
    """
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    dropped = []
    
    for i, cell in enumerate(cells):
        # Strip loading section
        if nb_path == 'FABQ-RC-Dense-27B-Notebook.ipynb':
            if 28 <= i <= 32:  # Loading section + workaround + 3.7 Verify + Memory-Optimized
                dropped.append(f"Dropped cell {i}: loading/reload section")
                continue
        elif nb_path == 'Main-FABQ-RC-Notebook.ipynb':
            if 28 <= i <= 34:  # Loading + workaround + HfApi upload + VRAM monitoring + metadata check
                dropped.append(f"Dropped cell {i}: loading/reload section")
                continue
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return dropped, len(nb['cells'])

print("=== Stripping loading section ===")
d1, c1 = strip_loading_section('FABQ-RC-Dense-27B-Notebook.ipynb')
for x in d1:
    print(f"  {x}")
print(f"Result: {c1} cells\n")

d2, c2 = strip_loading_section('Main-FABQ-RC-Notebook.ipynb')
for x in d2:
    print(f"  {x}")
print(f"Result: {c2} cells")

print("\nDone.")