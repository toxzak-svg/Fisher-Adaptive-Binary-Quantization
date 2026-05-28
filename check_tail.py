import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
lines = []
lines.append(f"Total cells: {len(nb['cells'])}")
for i in range(33, len(nb['cells'])):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    lines.append(f"Cell {i}: len={len(src)}, first100={repr(src[:100])}")
open('cells_33_to_end.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print("Done")