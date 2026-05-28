import json

def verify_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    issues = []
    
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        
        # Check for stale 35B references
        if '35B' in src and 'Qwen' in src.upper():
            issues.append(f"Cell {i}: Still has 35B reference: {src[:100]}")
        
        # Check for wrong INT8 vs INT4
        if 'INT8_FRACTION' in src and nb_path == 'FABQ-RC-Dense-27B-Notebook.ipynb':
            issues.append(f"Cell {i}: Still has INT8_FRACTION (should be INT4_FRACTION)")
        
        if 'INT4_FRACTION' in src and 'int8' in src.lower():
            issues.append(f"Cell {i}: INT4_FRACTION but comment says int8")
        
        # Check for wrong blocksize candidates
        if 'BS_CANDIDATES = [16, 32' in src:
            issues.append(f"Cell {i}: Old BS_CANDIDATES with 16,32 found")
        
        # Check for duplicate consecutive import gc
        if i > 0:
            prev = ''.join(cells[i-1].get('source', []))
            if src.strip() == 'import gc' and prev.strip() == 'import gc':
                issues.append(f"Cell {i}: Consecutive duplicate import gc")
    
    return issues

print("=== FABQ-RC-Dense-27B-Notebook.ipynb ===")
issues_27b = verify_notebook('FABQ-RC-Dense-27B-Notebook.ipynb')
if issues_27b:
    for issue in issues_27b:
        print(f"  ISSUE: {issue}")
else:
    print("  No issues found!")

print()
print("=== Main-FABQ-RC-Notebook.ipynb ===")
issues_main = verify_notebook('Main-FABQ-RC-Notebook.ipynb')
if issues_main:
    for issue in issues_main:
        print(f"  ISSUE: {issue}")
else:
    print("  No issues found!")

# Also check cell count
for nb_name in ['FABQ-RC-Dense-27B-Notebook.ipynb', 'Main-FABQ-RC-Notebook.ipynb']:
    nb = json.load(open(nb_name, encoding='utf-8'))
    print(f"\n{nb_name}: {len(nb['cells'])} cells")