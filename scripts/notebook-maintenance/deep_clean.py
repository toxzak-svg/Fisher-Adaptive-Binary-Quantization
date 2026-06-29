"""
Deep clean and verify FABQ-RC notebooks.
"""

import json

def deep_clean(nb_path, model_name="27B", expected_bs=None):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    changes = []
    
    # Check each cell for issues
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        
        # Check for int4/int8 issues
        if 'INT4_FRACTION' in src and 'int8' in src.lower():
            new_src = src.replace('# Keep top 5% Fisher channels as int8', '# Keep top 5% Fisher channels as int4')
            if new_src != src:
                cell['source'] = [new_src]
                changes.append(f"Cell {i}: Fixed int8 -> int4 in comment")
        
        # Check for 35B references in 27B notebook
        if model_name == "27B" and '35B' in src and 'Qwen' in src.upper():
            new_src = src.replace('Qwen3.6-35B-A3B', 'Qwen3.6-27B')
            new_src = new_src.replace('35B', '27B')
            if new_src != src:
                cell['source'] = [new_src]
                changes.append(f"Cell {i}: Fixed 35B -> 27B model reference")
        
        if model_name == "27B" and '### 3.1b Scaling to 35B Model' in src:
            new_src = src.replace('### 3.1b Scaling to 35B Model', '### 3.1b Scaling to 27B Model')
            cell['source'] = [new_src]
            changes.append(f"Cell {i}: Fixed section header 35B -> 27B")
    
    # Remove consecutive duplicate `import gc` and `import torch` cells
    new_cells = []
    prev_imports = set()
    for cell in cells:
        src = ''.join(cell.get('source', []))
        stripped = src.strip()
        
        # Detect single-import cells
        is_gc = stripped == 'import gc'
        is_torch = stripped == 'import torch'
        
        if is_gc and 'gc' in prev_imports:
            changes.append(f"Dropped duplicate import gc")
            continue
        if is_torch and 'torch' in prev_imports:
            changes.append(f"Dropped duplicate import torch")
            continue
        
        new_cells.append(cell)
        if is_gc:
            prev_imports.add('gc')
        if is_torch:
            prev_imports.add('torch')
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return changes

print("=== Deep Cleaning FABQ-RC-Dense-27B-Notebook.ipynb ===")
ch1 = deep_clean('FABQ-RC-Dense-27B-Notebook.ipynb', model_name="27B")
for c in ch1:
    print(f"  {c}")
print(f"Total: {len(ch1)} changes")

print()
print("=== Deep Cleaning Main-FABQ-RC-Notebook.ipynb ===")
ch2 = deep_clean('Main-FABQ-RC-Notebook.ipynb', model_name="27B")
for c in ch2:
    print(f"  {c}")
print(f"Total: {len(ch2)} changes")

print("\nDone.")