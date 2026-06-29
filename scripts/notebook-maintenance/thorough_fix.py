import json

def thorough_int4_fix(nb_path, cell_idx):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell = nb['cells'][cell_idx]
    src = ''.join(cell.get('source', []))
    
    # Thorough replacements
    src = src.replace('int8_fraction', 'int4_fraction')
    src = src.replace('n_int8', 'n_int4')
    src = src.replace('int8_channels', 'int4_channels')
    src = src.replace('"int8"', '"int4"')
    src = src.replace("'int8'", "'int4'")
    src = src.replace('int8 channels', 'int4 channels')
    
    cell['source'] = [src]
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return src

print("=== Thorough Fix ===")
r1 = thorough_int4_fix('FABQ-RC-Dense-27B-Notebook.ipynb', 17)
print("FABQ-RC-Dense-27B-Notebook.ipynb cell 17 done")
open('cell17_after.txt', 'w', encoding='utf-8').write(r1)

r2 = thorough_int4_fix('Main-FABQ-RC-Notebook.ipynb', 19)
print("Main-FABQ-RC-Notebook.ipynb cell 19 done")
open('cell19_after.txt', 'w', encoding='utf-8').write(r2)

print("Done")