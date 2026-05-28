import json

# Check if cells 35 and 36 are duplicates
nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))

cells_35_36 = ''.join(nb['cells'][35].get('source', []))
cells_36_37 = ''.join(nb['cells'][36].get('source', []))

lines = []
lines.append(f"Cell 35 len={len(cells_35_36)}")
lines.append(f"Cell 36 len={len(cells_36_37)}")
lines.append(f"Are they identical? {cells_35_36 == cells_36_37}")
lines.append("")
lines.append("Cell 35 first 200 chars:")
lines.append(repr(cells_35_36[:200]))
lines.append("")
lines.append("Cell 36 first 200 chars:")
lines.append(repr(cells_36_37[:200]))

# Also check cells 38, 42, 43 for standalone import patterns
for idx in [38, 42, 43]:
    src = ''.join(nb['cells'][idx].get('source', []))
    first_line = src.strip().split('\n')[0] if src.strip() else ''
    lines.append(f"Cell {idx} first line: {repr(first_line)}")

open('dup_check.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print("Done")