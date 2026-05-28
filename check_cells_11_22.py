import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
lines = []
for i in [11, 12, 13, 14, 15, 16, 17, 19, 22]:
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    open(f'cell_{i}_dense.txt', 'w', encoding='utf-8').write(src)
    lines.append(f"Cell {i}: len={len(src)}, first120={repr(src[:120])}")
open('dense_check.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print("Done")