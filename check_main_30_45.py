import json

nb = json.load(open('Main-FABQ-RC-Notebook.ipynb', encoding='utf-8'))
lines = []
for i in range(30, 45):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    lines.append(f"Cell {i}: len={len(src)}, first100={repr(src[:100])}")
open('main_30_45.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print("Done")

nb2 = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
lines2 = []
for i in range(28, 43):
    cell = nb2['cells'][i]
    src = ''.join(cell.get('source', []))
    lines2.append(f"Cell {i}: len={len(src)}, first100={repr(src[:100])}")
open('dense_28_43.txt', 'w', encoding='utf-8').write('\n'.join(lines2))
print("Done2")