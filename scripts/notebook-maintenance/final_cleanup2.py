import json

def final_cleanup(nb_path):
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
        lines = src.strip().split('\n')
        first_line = lines[0].strip() if lines else ''
        
        # Drop cell if it's ONLY 'import gc' and next cell is also 'import gc'
        if i < len(cells) - 1:
            next_src = ''.join(cells[i+1].get('source', []))
            next_first = next_src.strip().split('\n')[0].strip() if next_src.strip() else ''
            
            if first_line == 'import gc' and next_first == 'import gc':
                dropped.append(f"Dropped cell {i}: consecutive import gc")
                continue
        
        # Drop standalone matplotlib.pyplot import that's alone
        if first_line == 'import matplotlib.pyplot as plt':
            dropped.append(f"Dropped cell {i}: standalone matplotlib.pyplot import")
            continue
        
        new_cells.append(cell)
    
    nb['cells'] = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return dropped, len(nb['cells'])

print("=== Final cleanup ===")
d1, c1 = final_cleanup('FABQ-RC-Dense-27B-Notebook.ipynb')
for x in d1:
    print(f"  {x}")
print(f"Result: {c1} cells\n")

d2, c2 = final_cleanup('Main-FABQ-RC-Notebook.ipynb')
for x in d2:
    print(f"  {x}")
print(f"Result: {c2} cells")

print("\nDone.")