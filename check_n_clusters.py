import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))

# Check cell 22 - build_codebook function
src22 = ''.join(nb['cells'][22]['source'])
open('cell_22.txt', 'w', encoding='utf-8').write(src22)
print("Cell 22 n_clusters=256?", 'n_clusters=256' in src22)
print("Cell 22 first 500 chars:", repr(src22[:500]))
print()

# Check cell 23 - the call to build_codebook  
src23 = ''.join(nb['cells'][23]['source'])
open('cell_23.txt', 'w', encoding='utf-8').write(src23)
print("Cell 23 first 500 chars:", repr(src23[:500]))

# Check cell 19 - BS_PENALTIES
src19 = ''.join(nb['cells'][19]['source'])
open('cell_19.txt', 'w', encoding='utf-8').write(src19)
print()
print("Cell 19 BS_PENALTIES:", 'BS_PENALTIES' in src19)
print("Cell 19 first 500:", repr(src19[:500]))