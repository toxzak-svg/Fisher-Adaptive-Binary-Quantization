import json

def deep_clean_notebook(nb_path):
    """
    Remove the absurd download-reload cycle cells and clean up the notebook flow.
    The quantization pipeline should be:
    1. Load FP16 model
    2. Compute Fisher information
    3. Allocate precision (int4 vs binary)
    4. Quantize to FABQ-RC
    5. Save compressed to HF
    6. DONE - no need to reload!
    """
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    changes = []
    
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        
        # Cells that re-download from HF after quantization are BS
        if i in [35, 36]:
            changes.append(f"Dropped cell {i}: re-download from HF after quantization")
            continue
        
        # Drop standalone import gc cells
        stripped = src.strip()
        if stripped == 'import gc' or stripped == 'import gc\n':
            changes.append(f"Dropped cell {i}: standalone import gc")
            continue
        
        # Fix int8 -> int4 references in cell 47 (precision heatmap)
        if "'int8'" in src or '"int8"' in src or 'int8_count' in src:
            src = src.replace("'int8'", "'int4'")
            src = src.replace('"int8"', '"int4"')
            src = src.replace('int8_count', 'int4_count')
            src = src.replace('Int8 %', 'Int4 %')
            cell['source'] = [src]
            changes.append(f"Fixed cell {i}: int8 -> int4")
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return changes

print("=== Deep Clean FABQ-RC-Dense-27B-Notebook.ipynb ===")
ch1 = deep_clean_notebook('FABQ-RC-Dense-27B-Notebook.ipynb')
for c in ch1:
    print(f"  {c}")

print(f"\nTotal: {len(ch1)} changes")

print("\n=== Deep Clean Main-FABQ-RC-Notebook.ipynb ===")
ch2 = deep_clean_notebook('Main-FABQ-RC-Notebook.ipynb')
for c in ch2:
    print(f"  {c}")

print(f"\nTotal: {len(ch2)} changes")
print("\nDone.")