import json

for nb_name in ['FABQ-RC-Dense-27B-Notebook.ipynb', 'Main-FABQ-RC-Notebook.ipynb']:
    nb = json.load(open(nb_name, encoding='utf-8'))
    lines = []
    lines.append(f'=== {nb_name} ===')
    lines.append(f'Total cells: {len(nb["cells"])}')
    for i, cell in enumerate(nb['cells']):
        ct = cell.get('cell_type', '?')
        src = ''.join(cell.get('source', []))
        first_line = src.strip().split('\n')[0][:80] if src.strip() else '(empty)'
        lines.append(f'  Cell {i}: {ct} - {first_line}')
    lines.append('')
    with open(f'summary_{nb_name.replace("-", "_").replace(".","_")}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Wrote summary for {nb_name}')