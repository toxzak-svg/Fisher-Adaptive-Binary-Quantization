import json

for nb_name in ['FABQ-RC-Dense-27B-Notebook.ipynb', 'Main-FABQ-RC-Notebook.ipynb']:
    nb = json.load(open(nb_name, encoding='utf-8'))
    lines = []
    lines.append(f'=== {nb_name}: {len(nb["cells"])} cells ===')
    for i, cell in enumerate(nb['cells']):
        ct = cell.get('cell_type', '?')
        src = ''.join(cell.get('source', []))
        first = src.strip().split('\n')[0][:70] if src.strip() else '(empty)'
        lines.append(f'  {i}: {ct} | {first}')
    lines.append('')
    open(f'clean_{nb_name.replace("-", "_").replace(".","_")}.txt', 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'Done {nb_name}')