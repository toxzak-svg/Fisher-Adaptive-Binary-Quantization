import json

for nb_name in ['FABQ-RC-Dense-27B-Notebook.ipynb', 'Main-FABQ-RC-Notebook.ipynb']:
    nb = json.load(open(nb_name, encoding='utf-8'))
    print(f"=== {nb_name} ===")
    for i in range(10, 16):
        cell = nb['cells'][i]
        src = ''.join(cell.get('source', []))
        open(f'check_{nb_name}_{i}.txt', 'w', encoding='utf-8').write(src)
        print(f"Cell {i}: len={len(src)}, first80={repr(src[:80])}")
    print()