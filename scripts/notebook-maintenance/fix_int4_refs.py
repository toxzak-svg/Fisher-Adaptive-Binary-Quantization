import json

def fix_int4_cell(nb_path, cell_idx, new_var_name, old_var_name, param_name):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell = nb['cells'][cell_idx]
    src = ''.join(cell.get('source', []))
    
    # Fix the function parameter name
    src = src.replace(f'def allocate_precision(fisher_dict, {param_name}=0.05):',
                     f'def allocate_precision(fisher_dict, {new_var_name.lower()}=0.05):')
    
    # Fix the variable reference in the function call
    src = src.replace(f'allocation = allocate_precision(fisher_scores, {old_var_name})',
                     f'allocation = allocate_precision(fisher_scores, {new_var_name})')
    
    cell['source'] = [src]
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return src[:200]

print("=== Fixing FABQ-RC-Dense-27B-Notebook.ipynb cell 17 ===")
result = fix_int4_cell('FABQ-RC-Dense-27B-Notebook.ipynb', 17, 'INT4_FRACTION', 'INT8_FRACTION', 'int8_fraction')
print(f"First 200 chars after fix: {repr(result)}")

print()
print("=== Fixing Main-FABQ-RC-Notebook.ipynb cell 19 ===")
result2 = fix_int4_cell('Main-FABQ-RC-Notebook.ipynb', 19, 'INT4_FRACTION', 'INT8_FRACTION', 'int8_fraction')
print(f"First 200 chars after fix: {repr(result2)}")

print("\nDone.")