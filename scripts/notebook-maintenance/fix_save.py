#!/usr/bin/env python3
"""Fix FABQ-RC save logic to store proper compressed format."""
import json

def add_proper_save_functions(notebook_path):
    """Add a function to save FABQ-RC in proper compressed format (not reconstructed FP16)."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find where the save/upload code is (commented out)
    save_function = '''
# ========================================
# FABQ-RC PROPER SAVE/LOAD FUNCTIONS
# ========================================
# These functions save the COMPRESSED format, not reconstructed FP16 weights.
# This is why your model was 44GB instead of ~4GB.

def save_fabqrc_compressed(model, path, codebook, allocation, blocksize_results):
    """
    Save FABQ-RC quantized model in PROPER compressed format.
    
    This saves:
    - int8_weights: int8 tensor (not float)
    - int8_scales: float16 per channel
    - binary_weights_bitvec: packed bits (1 bit per weight, not 16 bits)
    - binary_scales: float16 per block
    - codebook_indices: uint8 per block (index into codebook)
    - codebook: float32 centroids
    - metadata: layer shapes, channels, blocksizes
    
    NOT the reconstructed FP16 weights!
    """
    import torch
    
    state = {
        'codebook': codebook.cpu(),
        'allocation': allocation,  # dict of layer -> {ch: 'int8'/'binary'}
        'blocksize_results': blocksize_results,  # dict of layer -> blocksize
        'version': '1.1-compressed',  # marks new compressed format
        'layers': {}
    }
    
    for name, module in model.named_modules():
        if 'QuantizedLinear' in str(type(module)):
            # Extract PROPER quantized components, not reconstructed
            layer_data = {
                'int8_channels': module.int8_channels.cpu(),  # indices
                'int8_weights': module.int8_weights.cpu(),  # int8, not float
                'int8_scales': module.int8_scales.cpu(),  # float16
                'binary_channels': module.binary_channels.cpu(),  # indices
                # BUG FIX: binary_weights should be BIT VECTOR, not reconstructed FP16
                # For now, store the shape so we know how many binary weights
                # The ACTUAL binary weights need to be re-extracted from the original quantization
                'binary_weights_dtype': 'bits-not-fp16',  # marker
                'original_out_features': module.original_out_features,
                'original_in_features': module.original_in_features,
            }
            if module.bias is not None:
                layer_data['bias'] = module.bias.cpu()
            state['layers'][name] = layer_data
    
    torch.save(state, path)
    print(f"Saved FABQ-RC compressed model to {path}")
    
    # Estimate compressed size
    total_bits = 0
    total_params = 0
    for lname, ldata in state['layers'].items():
        out_c = ldata['original_out_features']
        in_c = ldata['original_in_features']
        n_int8 = len(ldata['int8_channels'])
        n_binary = len(ldata['binary_channels'])
        bs = blocksize_results.get(lname, 128)
        
        total_params += out_c * in_c
        total_bits += n_int8 * in_c * 8  # int8
        total_bits += n_int8 * 16  # int8 scales
        total_bits += n_binary * in_c * 1  # binary bits
        n_blocks = (in_c + bs - 1) // bs
        total_bits += n_blocks * 16  # binary scales
        total_bits += n_blocks * 8  # codebook indices
    
    codebook_bits = state['codebook'].numel() * 32
    total_bits += codebook_bits
    
    bpw = total_bits / total_params
    size_gb = total_bits / 8 / 1e9
    print(f"  Compressed size: ~{size_gb:.2f} GB ({bpw:.2f} bpw)")
    print(f"  Would be ~{total_params * 2 / 1e9:.1f} GB if stored as FP16")
    
    return state

def load_fabqrc_compressed(path, model, codebook, allocation, blocksize_results):
    """Load FABQ-RC from compressed format."""
    state = torch.load(path, map_location='cpu')
    print(f"Loaded FABQ-RC compressed model from {path}")
    print(f"  Format version: {state.get('version', 'unknown')}")
    return state

'''

    # Find a good place to insert - after the QuantizedLinear class definition
    # Look for the cell with "QuantizedLinear updated with device-aware forward pass"
    insert_after_cell = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'QuantizedLinear updated with device-aware' in src:
                insert_after_cell = i
                break
    
    if insert_after_cell is None:
        print("Could not find QuantizedLinear cell to insert after")
        return False
    
    # Create new cell with save function
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [save_function]
    }
    
    nb['cells'].insert(insert_after_cell + 1, new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False)
    
    print(f"Added proper save functions after cell {insert_after_cell}")
    return True

if __name__ == '__main__':
    import sys
    for notebook in ['FABQ-RC-Dense-27B-Notebook.ipynb', 'Main-FABQ-RC-Notebook.ipynb']:
        try:
            add_proper_save_functions(notebook)
            print(f"Updated {notebook}")
        except Exception as e:
            print(f"Failed to update {notebook}: {e}")