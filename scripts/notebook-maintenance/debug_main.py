import json

with open('Main-FABQ-RC-Notebook.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

print(f'File size: {len(content)} bytes')
print(f'google.colab in content: {"google.colab" in content}')
print(f'userdata in content: {"userdata" in content}')
print(f'os.environ.get in content: {"os.environ.get" in content}')

# Check for HF_TOKEN patterns
print(f'HF_TOKEN in content: {"HF_TOKEN" in content}')

# Find HF_TOKEN occurrences
idx = 0
count = 0
while True:
    idx = content.find('HF_TOKEN', idx)
    if idx == -1:
        break
    count += 1
    print(f'HF_TOKEN at {idx}: {repr(content[idx-20:idx+50])}')
    idx += 1

print(f'\nTotal HF_TOKEN occurrences: {count}')