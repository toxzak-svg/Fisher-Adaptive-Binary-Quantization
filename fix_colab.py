import json

# Fix Main-FABQ-RC-Notebook.ipynb
print("Fixing Main-FABQ-RC-Notebook.ipynb...")
with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all occurrences
idx = 0
count = 0
while True:
    idx = content.find('google.colab', idx)
    if idx == -1:
        break
    count += 1
    print(f"Found at index {idx}: {repr(content[idx-20:idx+50])}")
    idx += 1

print(f"\nTotal occurrences: {count}")

# Remove the import lines
content_modified = content.replace('from google.colab import userdata\\n', '')
content_modified = content_modified.replace('from google.colab import userdata', '')

with open('Main-FABQ-RC-Notebook.ipynb', 'w', encoding='utf-8') as f:
    f.write(content_modified)

# Verify
with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    content2 = f.read()
print(f'After removal - google.colab in content: {"google.colab" in content2}')
print(f'userdata.get still present: {"userdata.get" in content2}')

# Replace userdata.get calls with os.environ.get
content3 = content2.replace("userdata.get('HF_TOKEN')", "os.environ.get('HF_TOKEN', 'YOUR_TOKEN_HERE')")

with open('Main-FABQ-RC-Notebook.ipynb', 'w', encoding='utf-8') as f:
    f.write(content3)

# Final verify
with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    content4 = f.read()
print(f'After userdata.get replacement - userdata.get present: {"userdata.get" in content4}')

print("\nMain-FABQ-RC-Notebook.ipynb fixed!")