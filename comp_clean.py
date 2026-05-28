import json

def comprehensive_cleanup(nb_path, is_27b_dense=False):
    """Clean both notebooks comprehensively."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    changes = []
    
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        
        # Fix 35B -> 27B references in 27B Dense notebook
        if is_27b_dense:
            if 'Qwen3.6-35B-A3B-FABQ-RC' in src:
                src = src.replace('Qwen3.6-35B-A3B-FABQ-RC', 'Qwen3.6-27B-FABQ-RC')
                cell['source'] = [src]
                changes.append(f"Cell {i}: Fixed 35B -> 27B repo reference")
            if 'Qwen3.6-35B-A3B' in src:
                src = src.replace('Qwen3.6-35B-A3B', 'Qwen3.6-27B')
                cell['source'] = [src]
                changes.append(f"Cell {i}: Fixed 35B -> 27B model reference")
            if 'toxzak/Qwen3.6-35B-A3B-FABQ-RC' in src:
                src = src.replace('toxzak/Qwen3.6-35B-A3B-FABQ-RC', 'toxzak/Qwen3.6-27B-FABQ-RC')
                cell['source'] = [src]
                changes.append(f"Cell {i}: Fixed HF repo reference 35B -> 27B")
        
        # Fix Main notebook 35B section header
        if 'Main' in nb_path and '### 3.1b Scaling to 35B Model' in src:
            src = src.replace('### 3.1b Scaling to 35B Model', '### 3.1b Scaling to 27B Model')
            cell['source'] = [src]
            changes.append(f"Cell {i}: Fixed section header 35B -> 27B")
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return changes

print("=== Comprehensive Cleanup ===")
c1 = comprehensive_cleanup('FABQ-RC-Dense-27B-Notebook.ipynb', is_27b_dense=True)
for c in c1:
    print(f"  {c}")
print(f"Total: {len(c1)} changes\n")

c2 = comprehensive_cleanup('Main-FABQ-RC-Notebook.ipynb', is_27b_dense=False)
for c in c2:
    print(f"  {c}")
print(f"Total: {len(c2)} changes")

print("\nDone.")