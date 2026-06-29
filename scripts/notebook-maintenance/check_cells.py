import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
lines = []
for i in range(30, 45):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    lines.append(f"Cell {i}: len={len(src)}, first80={repr(src[:80])}")
open('cells_30_45.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print("Done")