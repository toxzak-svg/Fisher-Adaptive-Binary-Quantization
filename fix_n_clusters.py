import json

def fix_n_clusters_and_penalties(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes = []
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        
        # Fix n_clusters=256 -> n_clusters=64
        if 'n_clusters=256' in src:
            src = src.replace('n_clusters=256', 'n_clusters=64')
            cell['source'] = [src]
            changes.append(f"Cell {i}: n_clusters 256 -> 64")
        
        # Fix BS_PENALTIES to match new BS_CANDIDATES [64, 128, 256, 512]
        if 'BS_PENALTIES' in src and '{16:' in src:
            # Replace old penalties with new ones
            old_penalties = "BS_PENALTIES = {16:'1.2', 32:'1.1', 64:'1.0', 128:'0.95', 256:'0.9'}"
            new_penalties = "BS_PENALTIES = {64:'1.1', 128:'1.0', 256:'0.9', 512:'0.85'}"
            if old_penalties in src:
                src = src.replace(old_penalties, new_penalties)
            else:
                # Try to find and replace the pattern more generically
                src = src.replace("BS_PENALTIES = {16:'1.2', 32:'1.1', 64:'1.0', 128:'0.95', 256:'0.9'}", 
                                 "BS_PENALTIES = {64:'1.1', 128:'1.0', 256:'0.9', 512:'0.85'}")
            cell['source'] = [src]
            changes.append(f"Cell {i}: BS_PENALTIES updated")
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return changes

print("=== FABQ-RC-Dense-27B-Notebook.ipynb ===")
c1 = fix_n_clusters_and_penalties('FABQ-RC-Dense-27B-Notebook.ipynb')
for x in c1:
    print(f"  {x}")

print("\n=== Main-FABQ-RC-Notebook.ipynb ===")
c2 = fix_n_clusters_and_penalties('Main-FABQ-RC-Notebook.ipynb')
for x in c2:
    print(f"  {x}")

print("\nDone.")