import json

# Check Main notebook for same issues
nb = json.load(open('Main-FABQ-RC-Notebook.ipynb', encoding='utf-8'))

for i in range(30, 50):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    open(f'main_cell_{i}.txt', 'w', encoding='utf-8').write(src)
    print(f"Cell {i}: len={len(src)}, first100={repr(src[:100])}")