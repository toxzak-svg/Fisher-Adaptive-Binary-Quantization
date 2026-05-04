from gguf import GGUFWriter
import numpy as np
import os

path = os.path.join(os.getcwd(), 'test.gguf')
print(f'Testing GGUF write to: {path}')

w = GGUFWriter(path, 'mistral', use_temp_file=False)
print(f'Writer state after init: {w.state}')

w.add_tensor('test', np.zeros((10, 10), dtype=np.float32))
print(f'Writer state after add_tensor: {w.state}')

# Try flush first
try:
    w.flush()
    print('Flush succeeded')
except Exception as e:
    print(f'Flush error: {e}')

# Try write methods
try:
    w.write_header_to_file()
    print('write_header_to_file succeeded')
except Exception as e:
    print(f'write_header_to_file error: {e}')

try:
    w.write_kv_data_to_file()
    print('write_kv_data_to_file succeeded')
except Exception as e:
    print(f'write_kv_data_to_file error: {e}')

try:
    w.write_tensors_to_file()
    print('write_tensors_to_file succeeded')
except Exception as e:
    print(f'write_tensors_to_file error: {e}')

try:
    w.close()
    print('close succeeded')
except Exception as e:
    print(f'close error: {e}')

print(f'Writer state after close: {w.state}')
print(f'File exists: {os.path.exists(path)}')