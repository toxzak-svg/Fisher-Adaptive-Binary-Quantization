import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
lines = []
for i in range(33, 50):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    lines.append(f"Cell {i}: {repr(src[:100])}")
open('cells_33_to_49.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print("Done")