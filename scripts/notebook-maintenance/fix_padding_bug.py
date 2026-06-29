#!/usr/bin/env python3
import json, sys

try:
    with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
        content = f.read()
    print(f'File read, size: {len(content)} bytes')
    
    nb = json.loads(content)
    print(f'JSON parsed, cells: {len(nb.get("cells", []))}')
    
    # Find the target pattern
    for ci, cell in enumerate(nb['cells']):
        src = cell.get('source', [])
        src_str = ''.join(src)
        if 'for start in range(0, weights.shape[1], bs):' in src_str:
            print(f'Found pattern in cell {ci}')
            # Find the line index
            for si, line in enumerate(src):
                if 'for start in range(0, weights.shape[1], bs):' in line:
                    print(f'  at source line {si}')
                    # Modify the next line
                    if si + 1 < len(src):
                        old_line = src[si + 1]
                        if 'end = min(start + bs, weights.shape[1])' in old_line:
                            # Insert our padding check
                            new_lines = old_line + '''\n                    \n                    # Skip padded blocks - they skew centroid computation\n                    if end - start < bs:\n                        continue\n                    \n'''
                            src[si + 1] = new_lines
                            print(f'  Modified line {si + 1}')
            
            with open('Main-FABQ-RC-Notebook.ipynb', 'w') as f:
                json.dump(nb, f)
            print('Saved changes!')
            break

except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)