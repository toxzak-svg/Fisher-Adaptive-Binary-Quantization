"""Verify the generated .ipynb is well-formed and printable."""
import json
from pathlib import Path

NB = Path(__file__).parent / "FABQ-RC-v2-colab-test.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

print(f"file: {NB.name} ({NB.stat().st_size} bytes)")
print(f"format: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"kernelspec: {nb['metadata']['kernelspec']['name']}")
print(f"language: {nb['metadata']['language_info']['name']} {nb['metadata']['language_info']['version']}")
print(f"cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src_lines = len(c["source"])
    first_line = c["source"][0].rstrip() if c["source"] else "(empty)"
    print(f"  cell {i+1:2d}: {c['cell_type']:8s} - {src_lines:3d} lines - {first_line[:60]}")

# Quick sanity: first cell should be markdown, then 8 code cells, then markdown, then 2 code
assert nb['cells'][0]['cell_type'] == 'markdown', "first cell should be markdown"
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"\ncode cells: {len(code_cells)} (expected 9)")
assert len(code_cells) == 9, f"expected 9 code cells, got {len(code_cells)}"

print("\nOK")
