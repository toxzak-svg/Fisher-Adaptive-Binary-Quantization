import json

def dump_cells(nb_path, start, end):
    nb = json.load(open(nb_path, encoding='utf-8'))
    lines = []
    for i in range(start, min(end, len(nb['cells']))):
        cell = nb['cells'][i]
        ct = cell.get('cell_type', '?')
        src = ''.join(cell.get('source', []))
        lines.append(f'=== Cell {i} ({ct}) ===')
        lines.append(src[:500])
        lines.append('')
    open(f'cells_{start}_{end}.txt', 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'Dumped cells {start}-{end} to cells_{start}_{end}.txt')

dump_cells('FABQ-RC-Dense-27B-Notebook.ipynb', 28, 50)