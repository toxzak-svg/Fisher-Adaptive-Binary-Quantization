import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
cell = nb['cells'][17]
src = ''.join(cell.get('source', []))
open('cell17.txt', 'w', encoding='utf-8').write(src)
print("Cell 17 content written to cell17.txt")
print(f"Length: {len(src)}")
print(f"First 300 chars: {repr(src[:300])}")

print()
nb2 = json.load(open('Main-FABQ-RC-Notebook.ipynb', encoding='utf-8'))
cell2 = nb2['cells'][19]
src2 = ''.join(cell2.get('source', []))
open('cell19_main.txt', 'w', encoding='utf-8').write(src2)
print("Cell 19 Main content written to cell19_main.txt")
print(f"Length: {len(src2)}")
print(f"First 300 chars: {repr(src2[:300])}")