import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
for i in range(33, 50):
    cell = nb['cells'][i]
    ct = cell.get('cell_type', '?')
    src = ''.join(cell.get('source', []))
    open(f'cell_{i}.txt', 'w', encoding='utf-8').write(src)
    print(f"Cell {i} ({ct}): len={len(src)}, repr={repr(src[:80])}")