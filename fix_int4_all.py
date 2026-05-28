import json

def fix_int4_comprehensive(nb_path, cell_idx):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell = nb['cells'][cell_idx]
    src = ''.join(cell.get('source', []))
    
    # Fix: n_int8 -> n_int4
    src = src.replace('n_int8', 'n_int4')
    # Fix: 'int8' channel label -> 'int4'
    src = src.replace("'int8'", "'int4'")
    # Fix: int8 channels (in print statements) -> int4
    src = src.replace('int8 channels', 'int4 channels')
    
    cell['source'] = [src]
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return src[:300]

print("=== Comprehensive Fix FABQ-RC-Dense-27B-Notebook.ipynb cell 17 ===")
result = fix_int4_comprehensive('FABQ-RC-Dense-27B-Notebook.ipynb', 17)
print(f"First 300 chars: {repr(result)}")

print()
print("=== Comprehensive Fix Main-FABQ-RC-Notebook.ipynb cell 19 ===")
result2 = fix_int4_comprehensive('Main-FABQ-RC-Notebook.ipynb', 19)
print(f"First 300 chars: {repr(result2)}")

print("\nDone.")