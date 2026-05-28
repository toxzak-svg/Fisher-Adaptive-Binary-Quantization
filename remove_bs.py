import json

def remove_bs_cells(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    dropped = []
    
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        
        # Drop cell 35 and 36 (duplicate re-download cells in FABQ-RC-Dense-27B)
        if nb_path == 'FABQ-RC-Dense-27B-Notebook.ipynb':
            if i in [35, 36]:
                dropped.append(f"Dropped cell {i}: duplicate re-download cell")
                continue
        
        # Fix cell 47 in Dense-27B: still uses 'int8' instead of 'int4'
        if nb_path == 'FABQ-RC-Dense-27B-Notebook.ipynb' and i == 47:
            if "'int8'" in src or '"int8"' in src:
                src = src.replace("'int8'", "'int4'")
                src = src.replace('"int8"', '"int4"')
                src = src.replace('int8_count', 'int4_count')
                src = src.replace('Int8 %', 'Int4 %')
                cell['source'] = [src]
                dropped.append(f"Fixed cell {i}: int8 -> int4")
        
        # Drop standalone import gc cells (keep ones that have other code)
        stripped = src.strip()
        if stripped == 'import gc' or stripped == 'import gc\n':
            dropped.append(f"Dropped cell {i}: standalone import gc")
            continue
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return dropped

print("=== FABQ-RC-Dense-27B-Notebook.ipynb ===")
d1 = remove_bs_cells('FABQ-RC-Dense-27B-Notebook.ipynb')
for d in d1:
    print(f"  {d}")

print("\n=== Main-FABQ-RC-Notebook.ipynb ===")
# Main notebook doesn't have cells 35,36 like Dense does, but check cell 47
nb = json.load(open('Main-FABQ-RC-Notebook.ipynb', encoding='utf-8'))
if len(nb['cells']) > 47:
    src = ''.join(nb['cells'][47].get('source', []))
    if "'int8'" in src or '"int8"' in src:
        src = src.replace("'int8'", "'int4'")
        src = src.replace('"int8"', '"int4"')
        src = src.replace('int8_count', 'int4_count')
        src = src.replace('Int8 %', 'Int4 %')
        nb['cells'][47]['source'] = [src]
        with open('Main-FABQ-RC-Notebook.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("  Fixed cell 47: int8 -> int4")
    else:
        print("  Cell 47: no int8 references")
else:
    print("  Cell 47 doesn't exist")

print("\nDone.")