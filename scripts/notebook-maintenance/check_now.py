import json

nb = json.load(open('FABQ-RC-Dense-27B-Notebook.ipynb', encoding='utf-8'))
src = ''.join(nb['cells'][17]['source'])
open('cell17_now.txt', 'w', encoding='utf-8').write(src)
print("Written")
print("'int8' in src:", ('int8' in src))
print("'int4' in src:", ('int4' in src))
print("First 500 chars:", repr(src[:500]))